import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np


# ====== Configure your classes here (id, name, color) ======
CLASS_CONFIG = [
    (0, "car",   "deepskyblue"),
    (1, "bus",   "lime green"),
    (2, "truck", "orange"),
    (3, "motorbike", "red")
]
CLASS_ID_TO_NAME = {cid: name for cid, name, _ in CLASS_CONFIG}
CLASS_ID_TO_COLOR = {cid: color for cid, _, color in CLASS_CONFIG}
DEFAULT_CLASS_ID = CLASS_CONFIG[0][0]


class BBoxEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bounding Box Editor (Multi-class)")

        self.canvas_width, self.canvas_height = 1280, 720
        self.yolo = YOLO("1classv4.pt")  

        default_start_dir = "/workspace/home/ambardi/mnt/gamma_raid/ambardi/Cars/"
        self.image_dir = filedialog.askdirectory(initialdir=default_start_dir, title="Select Folder with images/")
        if not self.image_dir:
            self.root.destroy()
            return

        self.image_list = sorted([
            f for f in os.listdir(os.path.join(self.image_dir, "images"))
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        if not self.image_list:
            messagebox.showerror("No images", "No images found in images/ folder.")
            self.root.destroy()
            return

        self.image_index = 0

        self.bboxes = []
        self.selected_idx = None
        self.click_start = None
        self.resize_dir = None
        self.resizing = False
        self.temp_box = None  # 
        self.current_class_id = DEFAULT_CLASS_ID

        self.setup_ui()
        self.load_image()
        self.root.mainloop()

    # ---------- UI ----------
    def setup_ui(self):
        self.left = tk.Frame(self.root)
        self.left.pack(side=tk.LEFT)
        self.right = tk.Frame(self.root)
        self.right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Canvas
        self.canvas = tk.Canvas(self.left, width=self.canvas_width, height=self.canvas_height, cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Class buttons / legend
        tk.Label(self.right, text="Classes", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 4))
        class_bar = tk.Frame(self.right)
        class_bar.pack(anchor="w", pady=(0, 6))

        for cid, name, color in CLASS_CONFIG:
            btn = tk.Button(class_bar, text=f"{cid}:{name}", relief=tk.RAISED, bg=color,
                            command=lambda c=cid: self.set_current_class(c))
            btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.current_class_label = tk.Label(self.right, text=f"Current class: {self._class_label(DEFAULT_CLASS_ID)}")
        self.current_class_label.pack(anchor="w", pady=(0, 6))

        # Tree with class + coords
        self.tree = ttk.Treeview(self.right, columns=("Class", "Coords"), show="headings", height=24)
        self.tree.heading("Class", text="Class")
        self.tree.heading("Coords", text="BBox (x1,y1,x2,y2)")
        self.tree.column("Class", width=90, anchor="center")
        self.tree.column("Coords", width=280, anchor="w")
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_tree)

        # Status + controls
        self.status = tk.Label(self.right, text="")
        self.status.pack(anchor="w", pady=(6, 0))

        nav = tk.Frame(self.right)
        nav.pack(anchor="w", pady=6)
        tk.Button(nav, text="Prev (←)", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Next (→)", command=self.next_image).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Save (Enter)", command=self.save_labels).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Delete (Backspace)", command=self.delete_selected_box).pack(side=tk.LEFT, padx=2)

        # Keybinds
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Return>", lambda e: self.save_labels())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected_box())

        # Number keys map to class ids if available (0–9)
        for num in "0123456789":
            self.root.bind(num, self._num_key_as_class)

        # Legend
        tk.Label(self.right, text="Legend", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(8, 0))
        legend = tk.Frame(self.right)
        legend.pack(anchor="w", pady=(2, 0))
        tk.Label(legend, text="Click & drag: draw / move; corners: resize.\n"
                              "Select row to focus box; numbers set current class.",
                 justify="left").pack(anchor="w")

    def _num_key_as_class(self, event):
        try:
            cid = int(event.char)
        except Exception:
            return
        if cid in CLASS_ID_TO_NAME:
            self.set_current_class(cid)

    def set_current_class(self, cid):
        self.current_class_id = cid
        self.current_class_label.config(text=f"Current class: {self._class_label(cid)}")

    def _class_label(self, cid):
        return f"{cid}:{CLASS_ID_TO_NAME.get(cid, 'unknown')}"

    # ---------- Image I/O & geometry ----------
    def load_image(self):
        img_name = self.image_list[self.image_index]
        img_path = os.path.join(self.image_dir, "images", img_name)
        self.original_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.original_image is None:
            messagebox.showerror("Error", f"Failed to read image: {img_path}")
            return
        h, w = self.original_image.shape[:2]

        # Compute resize & padding to fit canvas while preserving aspect ratio
        scale = min(self.canvas_width / w, self.canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        self.scale_x, self.scale_y = w / new_w, h / new_h  # canvas->orig: x*scale_x
        self.pad_x = (self.canvas_width - new_w) // 2
        self.pad_y = (self.canvas_height - new_h) // 2

        self.padded_image = np.full((self.canvas_height, self.canvas_width, 3), 128, dtype=np.uint8)
        resized = cv2.resize(self.original_image, (new_w, new_h))
        self.padded_image[self.pad_y:self.pad_y+new_h, self.pad_x:self.pad_x+new_w] = resized
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(self.padded_image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Reset bboxes
        self.bboxes.clear()
        self.selected_idx = None

        # Load labels if exist, else try model to seed boxes (assigned default class)
        label_file = os.path.join(self.image_dir, "labels", os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = parts
                    try:
                        cls_id = int(cls_id)
                        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                    except Exception:
                        continue
                    # Convert YOLO normalized -> original px
                    x1o = (cx - bw / 2.0) * w
                    y1o = (cy - bh / 2.0) * h
                    x2o = (cx + bw / 2.0) * w
                    y2o = (cy + bh / 2.0) * h
                    # Original -> canvas coords
                    x1 = int(x1o / self.scale_x) + self.pad_x
                    y1 = int(y1o / self.scale_y) + self.pad_y
                    x2 = int(x2o / self.scale_x) + self.pad_x
                    y2 = int(y2o / self.scale_y) + self.pad_y
                    self.bboxes.append([x1, y1, x2, y2, cls_id])
        else:
            results = self.yolo(img_path, verbose=False)[0]
            for box in results.boxes:
                x1o, y1o, x2o, y2o = map(float, box.xyxy[0])
                x1 = int(x1o / self.scale_x) + self.pad_x
                y1 = int(y1o / self.scale_y) + self.pad_y
                x2 = int(x2o / self.scale_x) + self.pad_x
                y2 = int(y2o / self.scale_y) + self.pad_y
                self.bboxes.append([x1, y1, x2, y2, DEFAULT_CLASS_ID])

        self.redraw()

    def redraw(self):
        self.canvas.delete("box")
        self.canvas.delete("handle")
        self.canvas.delete("highlight")
        self.tree.delete(*self.tree.get_children())

        for i, (x1, y1, x2, y2, cls_id) in enumerate(self.bboxes):
            base_color = CLASS_ID_TO_COLOR.get(cls_id, "white")
            outline_color = "red" if i == self.selected_idx else base_color

            # Slight translucent highlight if selected (draw first so under handles)
            if i == self.selected_idx:
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=base_color, stipple="gray25", outline="", tags="highlight"
                )

            # Main outline
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2, tags="box")

            # Corner handles (selected)
            if i == self.selected_idx:
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                                 fill="yellow", outline="", tags="handle")

            # Tree row (id is index)
            self.tree.insert(
                "", tk.END, iid=str(i),
                values=(self._class_label(cls_id), f"{x1},{y1},{x2},{y2}")
            )

        # Temp green dashed box (for drawing new)
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", dash=(4, 2), width=2, tags="box")

        self.status.config(text=f"Image {self.image_index + 1} / {len(self.image_list)}    "
                                f"Boxes: {len(self.bboxes)}")

    # ---------- Mouse & selection ----------
    def get_handle_at(self, x, y, margin=6):
        if self.selected_idx is None:
            return None
        x1, y1, x2, y2, _ = self.bboxes[self.selected_idx]
        handles = {'tl': (x1, y1), 'tr': (x2, y1), 'bl': (x1, y2), 'br': (x2, y2)}
        for name, (hx, hy) in handles.items():
            if abs(x - hx) <= margin and abs(y - hy) <= margin:
                return name
        return None

    def on_click(self, e):
        self.click_start = (e.x, e.y)
        self.resize_dir = self.get_handle_at(e.x, e.y)
        if self.resize_dir:
            self.resizing = True
        else:
            self.selected_idx = self.find_nearest_box(e.x, e.y)
            self.resizing = False
            if self.selected_idx is None:
                # start drawing a new box with current class
                self.temp_box = [e.x, e.y, e.x, e.y]
        self.redraw()

    def on_drag(self, e):
        if self.click_start is None:
            return

        if self.selected_idx is not None:
            x1, y1, x2, y2, cls_id = self.bboxes[self.selected_idx]
            dx = e.x - self.click_start[0]
            dy = e.y - self.click_start[1]

            if self.resizing:
                if self.resize_dir == 'tl':
                    x1 += dx; y1 += dy
                elif self.resize_dir == 'tr':
                    x2 += dx; y1 += dy
                elif self.resize_dir == 'bl':
                    x1 += dx; y2 += dy
                elif self.resize_dir == 'br':
                    x2 += dx; y2 += dy
            else:
                x1 += dx; x2 += dx
                y1 += dy; y2 += dy

            self.bboxes[self.selected_idx] = [x1, y1, x2, y2, cls_id]
            self.click_start = (e.x, e.y)

        elif self.temp_box:
            x0, y0 = self.click_start
            self.temp_box = [min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)]

        self.redraw()

    def on_release(self, e):
        if self.temp_box:
            x1, y1, x2, y2 = self.temp_box
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.bboxes.append([x1, y1, x2, y2, self.current_class_id])
                self.selected_idx = len(self.bboxes) - 1
        self.temp_box = None
        self.click_start = None
        self.resizing = False
        self.resize_dir = None
        self.redraw()

    def find_nearest_box(self, x, y):
        min_dist = float('inf')
        nearest_idx = None
        for i, (x1, y1, x2, y2, _) in enumerate(self.bboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = (x - cx) ** 2 + (y - cy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        return nearest_idx

    def on_select_tree(self, e):
        sel = self.tree.selection()
        if sel:
            self.selected_idx = int(sel[0])
            self.redraw()

    def delete_selected_box(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.bboxes):
            del self.bboxes[self.selected_idx]
            self.selected_idx = None
            self.redraw()

    # ---------- Save / Prev / Next ----------
    def save_labels(self):
        if not self.bboxes:
            # still allow saving empty file to clear labels
            pass

        img_name = os.path.splitext(self.image_list[self.image_index])[0]
        label_dir = os.path.join(self.image_dir, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, f'{img_name}.txt')

        h, w = self.original_image.shape[:2]
        with open(label_path, 'w') as f:
            for x1, y1, x2, y2, cls_id in self.bboxes:
                # canvas -> original px
                x1o = (x1 - self.pad_x) * self.scale_x
                y1o = (y1 - self.pad_y) * self.scale_y
                x2o = (x2 - self.pad_x) * self.scale_x
                y2o = (y2 - self.pad_y) * self.scale_y
                # convert to YOLO normalized
                cx = ((x1o + x2o) / 2.0) / w
                cy = ((y1o + y2o) / 2.0) / h
                bw = abs(x2o - x1o) / w
                bh = abs(y2o - y1o) / h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        messagebox.showinfo("Saved", f"Saved to {label_path}")

    def next_image(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.load_image()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()


if __name__ == "__main__":
    BBoxEditor()
