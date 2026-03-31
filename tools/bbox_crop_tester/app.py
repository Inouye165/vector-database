"""BBox Crop Tester GUI application for YOLO object detection testing."""
from __future__ import annotations

import csv
import json
import os
import threading
import traceback
from pathlib import Path
from threading import Event
from tkinter import (
    BOTH,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    Scrollbar,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)
from tkinter import ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk

# UI Constants for better maintainability
PREVIEW_MAX_WIDTH = 900
PREVIEW_MAX_HEIGHT = 620
THUMBNAIL_SIZE = 120
MIN_PREVIEW_WIDTH = 320
BBOX_LINE_WIDTH = 3
BBOX_TEXT_OFFSET = 24
BBOX_TEXT_MARGIN = 6
LETTERBOX_COLOR = (245, 245, 245)

try:
    from detector_improved import (
        detect_and_crop_folder_parallel as detect_and_crop_folder,
        DetectionConfig,
    )
    from config import PROFILES
    from io_utils import ensure_output_subdirs, save_manifest, setup_logging
    from models import ImageDetectionResult
except ImportError:
    from detector import detect_and_crop_folder
    DetectionConfig = None
from config import PROFILES
from io_utils import ensure_output_subdirs, save_manifest, setup_logging
from models import ImageDetectionResult


class BBoxCropTesterApp:
    """Main application class for BBox Crop Tester GUI.

    Provides a Tkinter-based interface for testing YOLO object detection
    and bounding box cropping functionality on image folders.
    """
    def __init__(self, root: Tk) -> None:
        """Initialize the BBox Crop Tester application.

        Args:
            root: Tkinter root window instance
        """
        self.root = root
        self.root.title("BBox Crop Tester")

        base_dir = Path(__file__).resolve().parent
        self.base_dir = base_dir
        self.logger = setup_logging(base_dir)

        self._init_variables()
        self._init_ui_constants()
        self._build_ui()
        self._setup_keyboard_shortcuts()
        self._setup_tooltips()
        self._apply_theme()

    def _init_variables(self) -> None:
        """Initialize application variables and state."""
        self.selected_folder = StringVar(value="")
        self.mode_var = StringVar(value="all")
        self.max_images_var = StringVar(value="10")
        self.confidence_var = StringVar(value="0.25")
        self.profile_var = StringVar(value="balanced")
        self.resume_var = StringVar(value="1")
        self.status_var = StringVar(value="Idle")
        self.cancel_event = Event()

        self.thumb_images: list[ImageTk.PhotoImage] = []
        self._max_thumbnails_in_memory = 50
        self._thumbnail_cache: dict[str, ImageTk.PhotoImage] = {}
        self.current_results: list[ImageDetectionResult] = []
        self.filter_var = StringVar(value="all")
        self.min_confidence_var = StringVar(value="0.0")
        self.dark_mode_var = StringVar(value="light")
        self.preferences_file = self.base_dir / "preferences.json"
        self.summary_label: Label | None = None
        self.gallery_canvas: Canvas | None = None
        self.gallery_inner: Frame | None = None
        self.gallery_window: int | None = None
        self._load_preferences()

    def _init_ui_constants(self) -> None:
        """Initialize UI constants and color schemes."""
        self.box_colors = [
            "#e53935",
            "#1e88e5",
            "#43a047",
            "#8e24aa",
            "#fb8c00",
            "#00897b",
            "#6d4c41",
            "#3949ab",
        ]

    def _build_folder_selection(self, parent: Frame) -> None:
        """Build folder selection controls."""
        folder_btn = Button(parent, text="Choose Folder", command=self.choose_folder)
        folder_btn.grid(row=0, column=0, sticky="w")
        self._add_tooltip(folder_btn, "Select folder containing images (Ctrl+O)")

        folder_label = Label(parent, textvariable=self.selected_folder, anchor="w")
        folder_label.grid(row=0, column=1, columnspan=4, sticky="w", padx=8)

    def _build_mode_controls(self, parent: Frame) -> None:
        """Build scan mode selection controls."""
        mode_label = Label(parent, text="Scan mode:")
        mode_label.grid(row=1, column=0, sticky="w", pady=(10, 0))

        all_radio = ttk.Radiobutton(
            parent, text="All images", variable=self.mode_var, value="all"
        )
        firstn_radio = ttk.Radiobutton(
            parent, text="First N images", variable=self.mode_var, value="firstn"
        )
        all_radio.grid(row=1, column=1, sticky="w", pady=(10, 0))
        firstn_radio.grid(row=1, column=2, sticky="w", pady=(10, 0))

        n_label = Label(parent, text="N:")
        n_label.grid(row=1, column=3, sticky="w", padx=(16, 0), pady=(10, 0))
        n_entry = Entry(parent, textvariable=self.max_images_var, width=6)
        n_entry.grid(row=1, column=4, sticky="w", pady=(10, 0))

    def _build_scan_controls(self, parent: Frame) -> None:
        """Build scan parameter and execution controls."""
        conf_label = Label(parent, text="Confidence threshold:")
        conf_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
        conf_entry = Entry(parent, textvariable=self.confidence_var, width=8)
        conf_entry.grid(row=2, column=1, sticky="w", pady=(10, 0))

        profile_label = Label(parent, text="Profile:")
        profile_label.grid(row=2, column=2, sticky="w", padx=(16, 0), pady=(10, 0))
        profile_combo = ttk.Combobox(
            parent,
            textvariable=self.profile_var,
            values=tuple(PROFILES.keys()),
            width=12,
            state="readonly",
        )
        profile_combo.grid(row=2, column=3, sticky="w", pady=(10, 0))
        profile_combo.bind("<<ComboboxSelected>>", self.on_profile_change)

        run_btn = Button(parent, text="Run scan", command=self.on_run_clicked)
        run_btn.grid(row=2, column=4, sticky="e", pady=(10, 0))
        self._add_tooltip(run_btn, "Start scanning images (Ctrl+R)")

        cancel_btn = Button(parent, text="Cancel", command=self.on_cancel_clicked)
        cancel_btn.grid(row=3, column=4, sticky="e", pady=(8, 0))
        self._add_tooltip(cancel_btn, "Cancel current scan (Ctrl+C)")
        resume_check = ttk.Checkbutton(
            parent,
            text="Resume after crash/cancel",
            variable=self.resume_var,
            onvalue="1",
            offvalue="0",
        )
        resume_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_filter_controls(self, parent: Frame) -> None:
        """Build result filtering controls."""
        filter_frame = Frame(parent)
        filter_frame.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(10, 0))

        Label(filter_frame, text="Filter by class:").pack(side=LEFT, padx=(0, 8))
        filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            values=("all", "person", "car", "truck", "bus",
                    "motorcycle", "bicycle", "dog", "cat", "other"),
            width=15,
            state="readonly",
        )
        filter_combo.pack(side=LEFT, padx=(0, 16))
        filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        Label(filter_frame, text="Min confidence:").pack(side=LEFT, padx=(0, 8))
        conf_entry = Entry(filter_frame, textvariable=self.min_confidence_var, width=6)
        conf_entry.pack(side=LEFT, padx=(0, 16))
        conf_entry.bind('<Return>', lambda e: self._on_filter_change())

        apply_btn = Button(filter_frame, text="Apply Filter", command=self._on_filter_change)
        apply_btn.pack(side=LEFT, padx=(0, 16))
        self._add_tooltip(apply_btn, "Apply filters to results (Ctrl+F)")

        export_btn = Button(filter_frame, text="Export Results", command=self._export_results)
        export_btn.pack(side=LEFT)
        self._add_tooltip(export_btn, "Export results to CSV/JSON (Ctrl+E)")

        # Add dark mode toggle
        dark_btn = Button(filter_frame, text="🌙", command=self._toggle_dark_mode, width=3)
        dark_btn.pack(side=RIGHT, padx=(8, 0))
        self._add_tooltip(dark_btn, "Toggle dark mode (Ctrl+D)")

    def _load_preferences(self) -> None:
        """Load user preferences from file."""
        try:
            if self.preferences_file.exists():
                with open(self.preferences_file, 'r',
                          encoding='utf-8') as f:
                    prefs = json.load(f)
                    self.dark_mode_var.set(prefs.get('dark_mode', 'light'))
                    self.confidence_var.set(prefs.get('confidence', '0.25'))
                    self.profile_var.set(prefs.get('profile', 'balanced'))
                    self.mode_var.set(prefs.get('mode', 'all'))
                    self.max_images_var.set(prefs.get('max_images', '10'))
                    self.selected_folder.set(prefs.get('last_folder', ''))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to load preferences: {e}")

    def _save_preferences(self) -> None:
        """Save current preferences to file."""
        try:
            prefs = {
                'dark_mode': self.dark_mode_var.get(),
                'confidence': self.confidence_var.get(),
                'profile': self.profile_var.get(),
                'mode': self.mode_var.get(),
                'max_images': self.max_images_var.get(),
                'last_folder': self.selected_folder.get()
            }
            with open(self.preferences_file, 'w',
                     encoding='utf-8') as f:
                json.dump(prefs, f, indent=2)
        except (OSError, TypeError) as e:
            self.logger.warning(f"Failed to save preferences: {e}")

    def _apply_theme(self) -> None:
        """Apply current theme to the application."""
        is_dark = self.dark_mode_var.get() == 'dark'

        if is_dark:
            # Dark mode colors
            bg_color = '#2b2b2b'
            fg_color = '#ffffff'
            button_bg = '#404040'
            entry_bg = '#353535'
            canvas_bg = '#2b2b2b'
        else:
            # Light mode colors
            bg_color = '#ffffff'
            fg_color = '#000000'
            button_bg = '#f0f0f0'
            entry_bg = '#ffffff'
            canvas_bg = '#ffffff'

        # Apply theme to root window
        self.root.configure(bg=bg_color)

        # Store theme colors for use in other widgets
        self.theme_colors = {
            'bg': bg_color,
            'fg': fg_color,
            'button_bg': button_bg,
            'entry_bg': entry_bg,
            'canvas_bg': canvas_bg
        }

        # Update all existing widgets
        self._update_widget_theme(self.root)

    def _update_widget_theme(self, widget) -> None:
        """Recursively update theme for all widgets."""
        try:
            widget_class = widget.winfo_class()

            if widget_class in ['Frame', 'Toplevel', 'Labelframe']:
                widget.configure(bg=self.theme_colors['bg'])
            elif widget_class == 'Label':
                widget.configure(bg=self.theme_colors['bg'], fg=self.theme_colors['fg'])
            elif widget_class == 'Button':
                widget.configure(bg=self.theme_colors['button_bg'], fg=self.theme_colors['fg'])
            elif widget_class == 'Entry':
                widget.configure(bg=self.theme_colors['entry_bg'], fg=self.theme_colors['fg'])
            elif widget_class == 'Canvas':
                widget.configure(bg=self.theme_colors['canvas_bg'])
        except (AttributeError, TypeError):
            pass  # Some widgets might not support these options

        # Recursively update children
        for child in widget.winfo_children():
            self._update_widget_theme(child)

    def _toggle_dark_mode(self) -> None:
        """Toggle between light and dark mode."""
        current_mode = self.dark_mode_var.get()
        new_mode = 'dark' if current_mode == 'light' else 'light'
        self.dark_mode_var.set(new_mode)
        self._apply_theme()
        self._save_preferences()

        # Update button text
        for widget in self.root.winfo_children():
            if isinstance(widget, Frame):
                for child in widget.winfo_children():
                    if isinstance(child, Frame):
                        for grandchild in child.winfo_children():
                            if hasattr(grandchild, 'cget') and grandchild.cget('text') in ['🌙', '☀️']:
                                grandchild.configure(text='☀️' if new_mode == 'dark' else '🌙')
                                return

    def _build_status_display(self) -> None:
        """Build status display area."""
        status_label = Label(self.root, textvariable=self.status_var, anchor="w")
        status_label.pack(side=TOP, fill=BOTH, padx=10, pady=(5, 0))

    def _build_results_display(self) -> None:
        """Build results display area with gallery."""
        results_frame = Frame(self.root)
        results_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)

        self.summary_label = Label(results_frame, text="No results yet")
        self.summary_label.pack(side=TOP, anchor="w")

        gallery_frame = Frame(results_frame)
        gallery_frame.pack(side=TOP, fill=BOTH, expand=True, pady=(5, 0))

        self.gallery_canvas = Canvas(gallery_frame, highlightthickness=0)
        self.gallery_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(gallery_frame, orient="vertical", command=self.gallery_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.gallery_canvas.configure(yscrollcommand=scrollbar.set)

        self.gallery_inner = Frame(self.gallery_canvas)
        self.gallery_window = self.gallery_canvas.create_window(
            (0, 0), window=self.gallery_inner, anchor="nw"
        )
        self.gallery_inner.bind("<Configure>", self._on_gallery_configure)
        self.gallery_canvas.bind("<Configure>", self._on_canvas_configure)
        self.on_profile_change()

    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts for common actions."""
        self.root.bind('<Control-o>', lambda e: self.choose_folder())
        self.root.bind('<Control-r>', lambda e: self.on_run_clicked())
        self.root.bind('<Control-c>', lambda e: self.on_cancel_clicked())
        self.root.bind('<Control-q>', lambda e: self._on_quit())
        self.root.bind('<Control-f>', lambda e: self._on_filter_change())
        self.root.bind('<Control-e>', lambda e: self._export_results())
        self.root.bind('<Control-d>', lambda e: self._toggle_dark_mode())
        self.root.bind('<F1>', lambda e: self._show_help())
        self.root.bind('<Escape>', lambda e: self.on_cancel_clicked())

    def _setup_tooltips(self) -> None:
        """Setup tooltips for UI elements."""
        # Tooltips will be added to individual widgets in their respective build methods

    def _add_tooltip(self, widget, text: str) -> None:
        """Add tooltip to a widget."""
        def on_enter(event):
            tooltip = Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = Label(tooltip, text=text, background="lightyellow",
                         relief="solid", borderwidth=1, font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    def _on_quit(self) -> None:
        """Handle application quit."""
        self._save_preferences()
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.quit()
            self.root.destroy()

    def _show_help(self) -> None:
        """Show help dialog with keyboard shortcuts."""
        help_text = """BBox Crop Tester - Help

Keyboard Shortcuts:
Ctrl+O - Choose folder
Ctrl+R - Run scan
Ctrl+C - Cancel scan
Ctrl+F - Apply filters
Ctrl+E - Export results
Ctrl+D - Toggle dark mode
Ctrl+Q - Quit application
F1 - Show this help
Escape - Cancel scan

Usage:
1. Select a folder containing images
2. Configure scan settings (mode, confidence, profile)
3. Click 'Run scan' to start detection
4. Filter results by class and confidence if needed
5. Export results using Ctrl+E
6. Toggle dark mode with Ctrl+D or 🌙 button

Profiles:
- Fast: Quick processing with YOLOv8n
- Balanced: Good accuracy with YOLOv8s
- High Recall: Maximum detection with YOLOv8m

Filtering:
- Filter by object class (person, car, etc.)
- Set minimum confidence threshold
- Filters apply to displayed results only

Features:
- Dark mode support (Ctrl+D)
- Preferences auto-save
- Export to CSV/JSON formats
- Keyboard shortcuts for all actions
- Tooltips on hover"""

        messagebox.showinfo("Help", help_text)

    def _build_ui(self) -> None:
        """Build the main user interface layout."""
        top_frame = Frame(self.root)
        top_frame.pack(side=TOP, fill=BOTH, padx=10, pady=10)

        self._build_folder_selection(top_frame)
        self._build_mode_controls(top_frame)
        self._build_scan_controls(top_frame)
        self._build_filter_controls(top_frame)
        self._build_status_display()
        self._build_results_display()

    def on_profile_change(self, _event: object | None = None) -> None:
        """Handle profile selection change."""
        profile = PROFILES[self.profile_var.get()]
        self.confidence_var.set(str(profile.confidence_threshold))

    def choose_folder(self) -> None:
        """Handle folder selection with validation and persistence."""
        # Start from last used folder if available
        initial_dir = self.selected_folder.get()
        if initial_dir and Path(initial_dir).exists():
            initial_dir = str(Path(initial_dir).parent)
        else:
            initial_dir = None

        folder = filedialog.askdirectory(title="Choose image folder", initialdir=initial_dir)
        if folder:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                self.status_var.set("Selected path is not a folder.")
                return
            if not os.access(folder, os.R_OK):
                self.status_var.set("No read permissions for selected folder.")
                return

            # Check if folder contains supported images
            image_files = [f for f in folder_path.iterdir()
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
            if not image_files:
                self.status_var.set("No supported image files found in folder.")
                return

            self.selected_folder.set(folder)
            self._save_preferences()  # Save immediately
            self.status_var.set(f"Selected: {folder_path.name}")

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self.cancel_event.set()
        self.status_var.set("Cancellation requested...")

    def on_run_clicked(self) -> None:
        folder = self.selected_folder.get().strip()
        if not folder:
            self.status_var.set("Please choose a folder.")
            return

        # Validate folder
        folder_path = Path(folder)
        if not folder_path.exists():
            self.status_var.set("Selected folder no longer exists.")
            return
        if not folder_path.is_dir():
            self.status_var.set("Selected path is not a folder.")
            return
        if not os.access(folder, os.R_OK | os.W_OK):
            self.status_var.set("Insufficient permissions for selected folder.")
            return

        # Validate confidence threshold
        try:
            conf = float(self.confidence_var.get())
        except ValueError:
            self.status_var.set(
                "Invalid confidence threshold. "
                "Please enter a number between 0.0 and 1.0.")
            return
        if conf < 0.0 or conf > 1.0:
            self.status_var.set("Confidence must be between 0.0 and 1.0.")
            return

        # Validate max images
        max_images: int | None
        if self.mode_var.get() == "firstn":
            try:
                n = int(self.max_images_var.get())
                if n <= 0:
                    self.status_var.set("N must be a positive integer.")
                    return
                max_images = n
            except ValueError:
                self.status_var.set("Invalid N value. Please enter a positive integer.")
                return
        else:
            max_images = None

        # Validate model file
        profile = PROFILES[self.profile_var.get()]
        model_path = self.base_dir / profile.model_path
        if not model_path.exists():
            self.status_var.set(f"Model file not found: {profile.model_path}")
            return

        input_dir = Path(folder)
        output_dir = self.base_dir

        self.status_var.set("Running scan...")
        self.summary_label.config(text="Processing...")
        self._clear_results()
        self.cancel_event.clear()

        thread = threading.Thread(
            target=self._run_scan_thread,
            args=(
                input_dir,
                output_dir,
                max_images,
                conf,
                profile.model_path,
                profile.enable_person_second_pass,
                profile.enable_tta_flip,
                self.resume_var.get() == "1",
            ),
            daemon=True,
        )
        thread.start()

    def _run_scan_thread(
        self,
        input_dir: Path,
        output_dir: Path,
        max_images: int | None,
        confidence: float,
        model_path: str,
        enable_person_second_pass: bool,
        enable_tta_flip: bool,
        resume_from_last: bool,
    ) -> None:
        try:
            ensure_output_subdirs(output_dir)
            self.logger.info(
                "Scan started",
                extra={
                    "event": "scan_started",
                    "image_path": str(input_dir),
                },
            )

            # Use improved detection config if available
            if DetectionConfig is not None:
                config = DetectionConfig(
                    confidence_threshold=confidence,
                    enable_person_second_pass=enable_person_second_pass,
                    enable_tta_flip=enable_tta_flip,
                    max_image_size=2048,
                    memory_threshold_mb=1024,
                    enable_nms=True,
                    nms_iou_threshold=0.45
                )
                batch = detect_and_crop_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    max_images=max_images,
                    config=config,
                    save_annotated=True,
                    save_crops=True,
                    progress_callback=self._on_progress_callback,
                    cancel_event=self.cancel_event,
                    resume_from_last=resume_from_last,
                )
            else:
                # Fallback to original function (no improved detector)
                batch = detect_and_crop_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    max_images=max_images,
                    confidence_threshold=confidence,
                    save_annotated=True,
                    save_crops=True,
                    progress_callback=self._on_progress_callback,
                    cancel_event=self.cancel_event,
                    resume_from_last=resume_from_last,
                    enable_person_second_pass=enable_person_second_pass,
                    enable_tta_flip=enable_tta_flip,
                )
            save_manifest(batch, output_dir)

            for err in batch.errors:
                self.logger.error(
                    err,
                    extra={
                        "event": "image_error",
                        "run_id": batch.run_id,
                        "image_path": str(input_dir),
                    },
                )

            self.logger.info(
                "Scan completed",
                extra={
                    "event": "scan_completed",
                    "run_id": batch.run_id,
                    "image_path": str(input_dir),
                },
            )

            self.root.after(
                0,
                self._update_results_ui,
                batch.images_scanned,
                batch.total_detections,
                batch.total_crops,
                batch.image_results,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            error_msg = str(exc)
            self.logger.error("Scan failed: %s\n%s", exc, traceback.format_exc())

            # Provide user-friendly error messages for common issues
            if "CUDA out of memory" in error_msg:
                user_msg = "GPU memory insufficient. Try using a smaller model or fewer images."
            elif "No space left on device" in error_msg:
                user_msg = "Disk space insufficient. Free up space and try again."
            elif "Permission denied" in error_msg:
                user_msg = "Permission denied. Check folder access rights."
            elif "File not found" in error_msg:
                user_msg = "Required file not found. Check model files and images."
            else:
                user_msg = error_msg

            self.root.after(0, self._set_error_status, user_msg)

    def _set_error_status(self, message: str) -> None:
        """Set error status message in UI."""
        self.status_var.set(f"Error: {message}")

    def _on_progress_callback(self, done: int, total: int, image_path: Path) -> None:
        self.root.after(
            0,
            lambda: self.status_var.set(f"Processing {done}/{total}: {image_path.name}"),
        )

    def _update_results_ui(
        self,
        images_scanned: int,
        total_detections: int,
        total_crops: int,
        image_results: list[ImageDetectionResult],
    ) -> None:
        self.status_var.set("Done.")
        self.summary_label.config(
            text=(
                f"Images scanned: {images_scanned} | "
                f"Detections: {total_detections} | "
                f"Crops saved: {total_crops}"
            )
        )

        self._clear_results()
        self.current_results = image_results
        filtered_results = self._apply_filters(image_results)
        for img_result in filtered_results:
            self._add_image_result_row(img_result)

    def _clear_results(self) -> None:
        for child in self.gallery_inner.winfo_children():
            child.destroy()
        self.thumb_images.clear()
        self._thumbnail_cache.clear()
        self.gallery_canvas.yview_moveto(0.0)

    def _on_gallery_configure(self, _event: object) -> None:
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

    def _on_canvas_configure(self, event: object) -> None:
        width = getattr(event, "width", None)
        if width:
            self.gallery_canvas.itemconfigure(self.gallery_window, width=width)

    def _add_image_result_row(self, img_result: ImageDetectionResult) -> None:
        container = Frame(self.gallery_inner, bd=1, relief="solid", padx=6, pady=6)
        container.pack(fill=X, pady=4)
        container.grid_columnconfigure(1, weight=1)

        # Show original image with in-memory overlays only (source file is never modified).
        preview_path = img_result.source_image

        image_panel = Frame(container, width=PREVIEW_MAX_WIDTH)
        image_panel.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        image_panel.grid_propagate(False)

        thumb_label = Label(image_panel, text="[preview unavailable]", anchor="center")
        thumb_label.pack(fill=BOTH, expand=True)

        # Load image efficiently
        try:
            # Get full-res EXIF-transposed dimensions (matches what the detector used).
            with Image.open(preview_path) as img:
                raw_disk_w, raw_disk_h = img.size
                # Determine orientation to know if width/height swap after transpose

            # Calculate if we need to load a smaller version for memory efficiency
            max_dimension = max(raw_disk_w, raw_disk_h)
            load_size = None
            if max_dimension > 2000:  # Load smaller version for very large images
                scale = 2000 / max_dimension
                load_size = (int(raw_disk_w * scale), int(raw_disk_h * scale))

            # Load image with appropriate size
            image = Image.open(preview_path)
            if load_size:
                image.thumbnail(load_size, Image.Resampling.LANCZOS)

            # Apply EXIF transpose - this creates canonical orientation
            image = ImageOps.exif_transpose(image).convert("RGB")
            canonical_w, canonical_h = image.size

            # Fit to full image inside panel while preserving original aspect ratio.
            display = image.copy()
            display.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.Resampling.LANCZOS)
            final_display_w, final_display_h = display.size

            # COMPREHENSIVE DEBUGGING - trace all coordinate spaces
            if img_result.detections:
                det = img_result.detections[0]  # First detection for debugging
                print(f"\n=== COMPLETE COORDINATE SPACE TRACE ===")
                print(f"Raw disk size: {raw_disk_w}x{raw_disk_h}")
                print(f"EXIF orientation: {1}")
                print(f"Expected full-res (post-transpose): {raw_disk_w}x{raw_disk_h}")
                print(f"Load size used: {load_size}")
                print(f"Canonical size (actual loaded): {canonical_w}x{canonical_h}")
                print(f"Final display size: {final_display_w}x{final_display_h}")
                print(f"Detector bbox: ({det.bbox.x1}, {det.bbox.y1}, {det.bbox.x2}, {det.bbox.y2})")

                # Check which coordinate space the detector bbox is actually in
                bbox_in_raw_space = (det.bbox.x1 <= raw_disk_w and det.bbox.x2 <= raw_disk_w and 
                                    det.bbox.y1 <= raw_disk_h and det.bbox.y2 <= raw_disk_h)
                bbox_in_full_res_space = (det.bbox.x1 <= raw_disk_w and det.bbox.x2 <= raw_disk_w and 
                                         det.bbox.y1 <= raw_disk_h and det.bbox.y2 <= raw_disk_h)
                bbox_in_canonical_space = (det.bbox.x1 <= canonical_w and det.bbox.x2 <= canonical_w and 
                                          det.bbox.y1 <= canonical_h and det.bbox.y2 <= canonical_h)

                print(f"Bbox fits in raw disk space: {bbox_in_raw_space}")
                print(f"Bbox fits in full-res space: {bbox_in_full_res_space}")
                print(f"Bbox fits in canonical space: {bbox_in_canonical_space}")

                # Draw color-coded boxes on preview shown in app only AFTER final resize.
                draw = ImageDraw.Draw(display)
                for idx, det in enumerate(img_result.detections):
                    color = self.box_colors[idx % len(self.box_colors)]
                    b = det.bbox

                    # Determine which coordinate space to use for scaling
                    if bbox_in_canonical_space:
                        scale_source_w, scale_source_h = canonical_w, canonical_h
                        coord_space_name = "canonical"
                    elif bbox_in_full_res_space:
                        scale_source_w, scale_source_h = raw_disk_w, raw_disk_h
                        coord_space_name = "full_res"
                    else:
                        scale_source_w, scale_source_h = raw_disk_w, raw_disk_h
                        coord_space_name = "raw_disk"

                    # Scale bbox from detected space to display
                    display_scale_x = final_display_w / scale_source_w
                    display_scale_y = final_display_h / scale_source_h

                    # Scale bbox for display
                    display_x1 = int(b.x1 * display_scale_x)
                    display_y1 = int(b.y1 * display_scale_y)
                    display_x2 = int(b.x2 * display_scale_x)
                    display_y2 = int(b.y2 * display_scale_y)

                    print(f"Using {coord_space_name} coordinate space for scaling")
                    print(f"Display scale factors: x={display_scale_x:.3f}, y={display_scale_y:.3f}")
                    print(f"Display bbox: ({display_x1}, {display_y1}, {display_x2}, {display_y2})")

                    # Use thin outline (2px) for better visibility
                    outline_width = 2
                    draw.rectangle([display_x1, display_y1, display_x2, display_y2], outline=color, width=outline_width)
                    draw.text(
                        (display_x1 + BBOX_TEXT_MARGIN, max(4, display_y1 - BBOX_TEXT_OFFSET)),
                        f"{idx + 1}:{det.label} {det.confidence:.2f}",
                        fill=color,
                    )

                    # DEBUG: Show crop bbox that was actually saved
                    if hasattr(det, 'crop_path') and det.crop_path:
                        print(f"Crop saved to: {det.crop_path}")
                        print(f"Crop uses detector bbox: ({b.x1}, {b.y1}, {b.x2}, {b.y2})")

                # Use the coordinate space that actually matches the bbox for "Original size" display
                if img_result.detections:
                    det = img_result.detections[0]
                    if det.bbox.x2 <= canonical_w and det.bbox.y2 <= canonical_h:
                        original_w, original_h = canonical_w, canonical_h
                        size_label = "canonical"
                    elif det.bbox.x2 <= raw_disk_w and det.bbox.y2 <= raw_disk_h:
                        original_w, original_h = raw_disk_w, raw_disk_h
                        size_label = "full_res"
                    else:
                        original_w, original_h = raw_disk_w, raw_disk_h
                        size_label = "raw_disk"
                    print(f"Using {size_label} size for UI display: {original_w}x{original_h}")
                else:
                    # Use canonical dimensions for consistent "Original size" display
                    original_w, original_h = canonical_w, canonical_h
            photo = ImageTk.PhotoImage(display)
            self.thumb_images.append(photo)
            thumb_label.configure(image=photo, text="")
            image_panel.configure(width=max(MIN_PREVIEW_WIDTH, display.width))
        except FileNotFoundError:
            self.logger.warning(f"Preview image not found: {preview_path}")
            thumb_label.configure(text="[image not found]")
        except PermissionError:
            self.logger.warning(f"Permission denied accessing: {preview_path}")
            thumb_label.configure(text="[access denied]")
        except (OSError, ValueError) as e:
            self.logger.error(f"Error loading preview {preview_path}: {e}")
            thumb_label.configure(text="[preview error]")

        info_frame = Frame(container)
        info_frame.grid(row=0, column=1, sticky="nsew")

        Label(
            info_frame,
            text=f"Source: {img_result.source_image.name}",
            anchor="w",
            justify="left",
        ).pack(fill=X)

        if "original_w" in locals() and "original_h" in locals():
            Label(
                info_frame,
                text=f"Original size: {original_w}x{original_h}",
                anchor="w",
                justify="left",
            ).pack(fill=X)

        if img_result.error:
            Label(
                info_frame,
                text=f"Error: {img_result.error}",
                fg="red",
                anchor="w",
                justify="left",
            ).pack(fill=X, pady=(2, 0))
            return

        Label(
            info_frame,
            text=f"Detections: {len(img_result.detections)}",
            anchor="w",
        ).pack(fill=X, pady=(4, 4))

        if not img_result.detections:
            Label(info_frame, text="No detections", anchor="w", justify="left").pack(fill=X)
            return

        for idx, det in enumerate(img_result.detections, start=1):
            det_row = Frame(info_frame, bd=1, relief="groove", padx=4, pady=4)
            det_row.pack(fill=X, pady=(0, 4))

            color = self.box_colors[(idx - 1) % len(self.box_colors)]

            color_chip = Label(det_row, text=" ", bg=color, width=2)
            color_chip.pack(side=LEFT, padx=(0, 6), fill=Y)

            crop_thumb = Label(
                det_row,
                text="[crop]",
                width=THUMBNAIL_SIZE,
                height=THUMBNAIL_SIZE,
                anchor="center",
                bd=1,
                relief="solid",
            )
            crop_thumb.pack(side=LEFT, padx=(0, 8))
            if det.crop_path:
                try:
                    # Use cached thumbnail if available
                    cache_key = str(det.crop_path)
                    if cache_key in self._thumbnail_cache:
                        crop_photo = self._thumbnail_cache[cache_key]
                    else:
                        crop_image = Image.open(det.crop_path)
                        crop_image = crop_image.convert("RGB")
                        crop_photo = ImageTk.PhotoImage(
                            self._build_letterboxed_thumbnail(crop_image, (THUMBNAIL_SIZE, THUMBNAIL_SIZE))
                        )

                        # Manage cache size
                        if len(self._thumbnail_cache) >= self._max_thumbnails_in_memory:
                            # Remove oldest entries
                            oldest_keys = list(self._thumbnail_cache.keys())[:10]
                            for key in oldest_keys:
                                del self._thumbnail_cache[key]

                        self._thumbnail_cache[cache_key] = crop_photo

                    self.thumb_images.append(crop_photo)
                    crop_thumb.configure(image=crop_photo, text="")
                except FileNotFoundError:
                    self.logger.warning(f"Crop image not found: {det.crop_path}")
                except PermissionError:
                    self.logger.warning(f"Permission denied accessing crop: {det.crop_path}")
                except (OSError, ValueError) as e:
                    self.logger.error(f"Error loading crop {det.crop_path}: {e}")

            bbox = det.bbox
            det_text = (
                f"{idx}. {det.label} ({det.confidence:.2f})\n"
                f"bbox=({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2})\n"
                f"size={bbox.width}x{bbox.height}"
            )
            Label(
                det_row,
                text=det_text,
                anchor="w",
                justify="left",
                fg=color,
            ).pack(side=LEFT, fill=X, expand=True)

    def _build_letterboxed_thumbnail(
        self, image: Image.Image, size: tuple[int, int]
    ) -> Image.Image:
        """Create a letterboxed thumbnail with proper aspect ratio preservation.

        Args:
            image: Source image to thumbnail
            size: Target size as (width, height)

        Returns:
            Letterboxed thumbnail image
        """
        target_w, target_h = size
        img = image.copy()
        img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), color=LETTERBOX_COLOR)
        offset_x = (target_w - img.width) // 2
        offset_y = (target_h - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        return canvas

    def _on_filter_change(self, _event: object | None = None) -> None:
        """Apply filters to current results."""
        if not self.current_results:
            self.status_var.set("No results to filter.")
            return

        try:
            min_conf = float(self.min_confidence_var.get())
            if min_conf < 0.0 or min_conf > 1.0:
                self.status_var.set("Confidence must be between 0.0 and 1.0.")
                return
        except ValueError:
            self.status_var.set("Invalid confidence value.")
            return

        self._clear_results()
        filtered_results = self._apply_filters(self.current_results)
        for img_result in filtered_results:
            self._add_image_result_row(img_result)

        self.status_var.set(f"Filtered: {len(filtered_results)} of {len(self.current_results)} results")

    def _apply_filters(self, results: list[ImageDetectionResult]) -> list[ImageDetectionResult]:
        """Apply current filters to results."""
        filtered = []
        filter_class = self.filter_var.get()
        min_conf = float(self.min_confidence_var.get())

        for img_result in results:
            if img_result.error:
                continue

            filtered_detections = []
            for det in img_result.detections:
                # Apply class filter
                if filter_class != "all" and det.label.lower() != filter_class:
                    continue

                # Apply confidence filter
                if det.confidence < min_conf:
                    continue

                filtered_detections.append(det)

            if filtered_detections:
                # Create new result with filtered detections
                filtered_result = ImageDetectionResult(
                    source_image=img_result.source_image,
                    detections=filtered_detections,
                    error=img_result.error
                )
                filtered.append(filtered_result)

        return filtered

    def _export_results(self) -> None:
        """Export current results to CSV or JSON."""
        if not self.current_results:
            messagebox.showwarning("Export", "No results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.json'):
                self._export_json(file_path)
            else:
                self._export_csv(file_path)

            messagebox.showinfo("Export", f"Results exported to {file_path}")
            self.status_var.set(f"Exported to {Path(file_path).name}")
        except (OSError, ValueError) as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
            self.status_var.set(f"Export failed: {str(e)}")

    def _export_csv(self, file_path: str) -> None:
        """Export results to CSV format."""
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'image', 'detection_id', 'label', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'width', 'height', 'crop_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for img_result in self.current_results:
                if img_result.error:
                    writer.writerow({
                        'image': img_result.source_image.name,
                        'detection_id': '',
                        'label': 'ERROR',
                        'confidence': '',
                        'x1': '', 'y1': '', 'x2': '', 'y2': '',
                        'width': '', 'height': '',
                        'crop_path': img_result.error
                    })
                    continue

                for idx, det in enumerate(img_result.detections, 1):
                    writer.writerow({
                        'image': img_result.source_image.name,
                        'detection_id': idx,
                        'label': det.label,
                        'confidence': f"{det.confidence:.3f}",
                        'x1': det.bbox.x1,
                        'y1': det.bbox.y1,
                        'x2': det.bbox.x2,
                        'y2': det.bbox.y2,
                        'width': det.bbox.width,
                        'height': det.bbox.height,
                        'crop_path': str(det.crop_path) if det.crop_path else ''
                    })

    def _export_json(self, file_path: str) -> None:
        """Export results to JSON format."""
        export_data = {
            'export_timestamp': Path(file_path).stem,
            'total_images': len(self.current_results),
            'images': []
        }

        for img_result in self.current_results:
            img_data = {
                'image_path': str(img_result.source_image),
                'error': img_result.error,
                'detections': []
            }

            if not img_result.error:
                for det in img_result.detections:
                    det_data = {
                        'label': det.label,
                        'confidence': det.confidence,
                        'bbox': {
                            'x1': det.bbox.x1,
                            'y1': det.bbox.y1,
                            'x2': det.bbox.x2,
                            'y2': det.bbox.y2,
                            'width': det.bbox.width,
                            'height': det.bbox.height
                        },
                        'crop_path': str(det.crop_path) if det.crop_path else None
                    }
                    img_data['detections'].append(det_data)

            export_data['images'].append(img_data)

        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)


def run_app() -> None:
    """Create and run the BBox Crop Tester application."""
    root = Tk()
    BBoxCropTesterApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()

