"""
Tkinter GUI for OhanaAI genealogical parent prediction system.
Provides user-friendly interface for file loading, visualization, and predictions.
"""

import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

import yaml

from .data_deduplication import DeduplicationEngine, DuplicateMatch
from .api.gedcom_parser import Family, Individual, parse_gedcom_file
from .predictor import OhanaAIPredictor, ParentPrediction

logger = logging.getLogger(__name__)


class OhanaAIGUI:
    """Main GUI application for OhanaAI."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.root = tk.Tk()
        self.root.title("OhanaAI - Genealogical Parent Prediction")
        self.root.geometry(
            f"{self.config['gui']['window_width']}x{self.config['gui']['window_height']}"
        )

        # Data storage
        self.gedcom_files: List[str] = []
        self.individuals: Dict[str, Individual] = {}
        self.families: Dict[str, Family] = {}
        self.predictions: List[ParentPrediction] = []
        self.duplicates: List[DuplicateMatch] = []

        # Models
        self.trainer: Optional[OhanaAITrainer] = None
        self.predictor: Optional[OhanaAIPredictor] = None
        self.dedup_engine: Optional[DeduplicationEngine] = None

        # Threading
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Setup GUI
        self._setup_gui()
        self._setup_logging()

        # Start checking for results
        self.root.after(100, self._check_results)

    def _setup_gui(self):
        """Setup the main GUI layout."""
        # Create menu bar
        self._create_menu_bar()

        # Create main frames
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - file management and controls
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # Right panel - visualization and results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup left panel
        self._setup_left_panel(left_frame)

        # Setup right panel
        self._setup_right_panel(right_frame)

        # Status bar
        self._setup_status_bar()

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Load GEDCOM Files...", command=self._load_gedcom_files
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save Predictions...", command=self._save_predictions
        )
        file_menu.add_command(label="Export GEDCOM...", command=self._export_gedcom)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Train Model", command=self._train_model)
        model_menu.add_command(label="Load Model...", command=self._load_model)
        model_menu.add_command(label="Run Predictions", command=self._run_predictions)

        # Deduplication menu
        dedup_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Deduplication", menu=dedup_menu)
        dedup_menu.add_command(
            label="Detect Duplicates", command=self._detect_duplicates
        )
        dedup_menu.add_command(
            label="Review Duplicates", command=self._review_duplicates
        )

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _setup_left_panel(self, parent):
        """Setup the left control panel."""
        # File section
        file_frame = ttk.LabelFrame(parent, text="GEDCOM Files", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        self.file_listbox = tk.Listbox(file_frame, height=6)
        self.file_listbox.pack(fill=tk.X, pady=(0, 5))

        file_buttons = ttk.Frame(file_frame)
        file_buttons.pack(fill=tk.X)

        ttk.Button(
            file_buttons, text="Add Files", command=self._load_gedcom_files
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            file_buttons, text="Remove", command=self._remove_selected_file
        ).pack(side=tk.LEFT)

        # Model section
        model_frame = ttk.LabelFrame(parent, text="Model Operations", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            model_frame, text="Train Model", command=self._train_model, width=20
        ).pack(pady=2, fill=tk.X)
        ttk.Button(
            model_frame, text="Load Model", command=self._load_model, width=20
        ).pack(pady=2, fill=tk.X)
        ttk.Button(
            model_frame, text="Run Predictions", command=self._run_predictions, width=20
        ).pack(pady=2, fill=tk.X)

        # Deduplication section
        dedup_frame = ttk.LabelFrame(parent, text="Deduplication", padding=5)
        dedup_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            dedup_frame,
            text="Detect Duplicates",
            command=self._detect_duplicates,
            width=20,
        ).pack(pady=2, fill=tk.X)
        ttk.Button(
            dedup_frame,
            text="Review Duplicates",
            command=self._review_duplicates,
            width=20,
        ).pack(pady=2, fill=tk.X)

        # Settings section
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding=5)
        settings_frame.pack(fill=tk.X, pady=(0, 5))

        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(
            value=self.config["gui"]["confidence_threshold"]
        )
        confidence_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
        )
        confidence_scale.pack(fill=tk.X, pady=2)

        self.confidence_label = ttk.Label(
            settings_frame, text=f"{self.confidence_var.get():.2f}"
        )
        self.confidence_label.pack(anchor=tk.W)
        confidence_scale.configure(command=self._update_confidence_label)

        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(parent, text="Status:").pack(anchor=tk.W, pady=(10, 0))
        self.progress_label = ttk.Label(parent, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(parent, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=5)

    def _setup_right_panel(self, parent):
        """Setup the right panel with tabs."""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Predictions tab
        pred_frame = ttk.Frame(notebook)
        notebook.add(pred_frame, text="Predictions")
        self._setup_predictions_tab(pred_frame)

        # Visualization tab
        vis_frame = ttk.Frame(notebook)
        notebook.add(vis_frame, text="Visualization")
        self._setup_visualization_tab(vis_frame)

        # Duplicates tab
        dup_frame = ttk.Frame(notebook)
        notebook.add(dup_frame, text="Duplicates")
        self._setup_duplicates_tab(dup_frame)

        # Log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Logs")
        self._setup_log_tab(log_frame)

    def _setup_predictions_tab(self, parent):
        """Setup the predictions display tab."""
        # Filter controls
        filter_frame = ttk.Frame(parent)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="Filter by confidence:").pack(side=tk.LEFT)
        self.filter_var = tk.DoubleVar(value=0.5)
        filter_scale = ttk.Scale(
            filter_frame,
            from_=0.0,
            to=1.0,
            variable=self.filter_var,
            orient=tk.HORIZONTAL,
            length=200,
        )
        filter_scale.pack(side=tk.LEFT, padx=5)
        filter_scale.configure(command=self._filter_predictions)

        # Predictions table
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Treeview for predictions
        columns = ("Child", "Candidate Parent", "Confidence", "Age Diff", "Constraints")
        self.pred_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=15
        )

        for col in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=150)

        # Scrollbars
        v_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.pred_tree.yview
        )
        h_scroll = ttk.Scrollbar(
            table_frame, orient=tk.HORIZONTAL, command=self.pred_tree.xview
        )
        self.pred_tree.configure(
            yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set
        )

        self.pred_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_visualization_tab(self, parent):
        """Setup the family tree visualization tab."""
        # Simple text-based visualization for now
        self.vis_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20)
        self.vis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            controls_frame, text="Show Family Tree", command=self._show_family_tree
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            controls_frame, text="Show Statistics", command=self._show_statistics
        ).pack(side=tk.LEFT, padx=5)

    def _setup_duplicates_tab(self, parent):
        """Setup the duplicates review tab."""
        # Duplicates table
        dup_columns = ("Individual 1", "Individual 2", "Similarity", "Reasons")
        self.dup_tree = ttk.Treeview(
            parent, columns=dup_columns, show="headings", height=15
        )

        for col in dup_columns:
            self.dup_tree.heading(col, text=col)
            self.dup_tree.column(col, width=200)

        self.dup_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls
        dup_controls = ttk.Frame(parent)
        dup_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            dup_controls, text="Merge Selected", command=self._merge_selected_duplicate
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(dup_controls, text="Ignore", command=self._ignore_duplicate).pack(
            side=tk.LEFT, padx=5
        )

    def _setup_log_tab(self, parent):
        """Setup the logging display tab."""
        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Clear button
        ttk.Button(
            parent, text="Clear Logs", command=lambda: self.log_text.delete(1.0, tk.END)
        ).pack(pady=5)

    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_logging(self):
        """Setup logging to display in GUI."""

        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget, queue_obj):
                super().__init__()
                self.text_widget = text_widget
                self.queue = queue_obj

            def emit(self, record):
                log_entry = self.format(record)
                self.queue.put(("log", log_entry))

        # Add GUI log handler
        gui_handler = GUILogHandler(self.log_text, self.result_queue)
        gui_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(gui_handler)

    def _update_confidence_label(self, value):
        """Update confidence threshold label."""
        self.confidence_label.config(text=f"{float(value):.2f}")

    def _load_gedcom_files(self):
        """Load GEDCOM files dialog."""
        try:
            print("Opening file dialog...")  # Debug print
            files = filedialog.askopenfilenames(
                title="Select GEDCOM Files",
                filetypes=[
                    ("GEDCOM files", "*.ged *.gedcom *.txt"),
                    ("All files", "*.*"),
                ],
            )
            print(f"Selected files: {files}")  # Debug print

            # Create uploads directory if it doesn't exist
            uploads_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "uploads"
            )
            os.makedirs(uploads_dir, exist_ok=True)

            for file in files:
                if file not in self.gedcom_files:
                    # Copy file to uploads directory
                    import shutil

                    filename = os.path.basename(file)
                    dest_path = os.path.join(uploads_dir, filename)

                    # Handle duplicate filenames by adding a number
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(dest_path):
                        new_filename = f"{base_name}_{counter}{ext}"
                        dest_path = os.path.join(uploads_dir, new_filename)
                        counter += 1

                    shutil.copy2(file, dest_path)
                    print(f"Copied {file} to {dest_path}")

                    # Use the copied file path
                    self.gedcom_files.append(dest_path)
                    self.file_listbox.insert(tk.END, os.path.basename(dest_path))

            self._update_status(
                f"Loaded and saved {len(files)} GEDCOM files to uploads folder"
            )

            # Auto-parse files
            if files:
                self._parse_gedcom_files()
        except Exception as e:
            print(f"Error in file dialog: {e}")  # Debug print
            messagebox.showerror("Error", f"Failed to open file dialog: {str(e)}")

    def _remove_selected_file(self):
        """Remove selected file from list."""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.file_listbox.delete(index)
            del self.gedcom_files[index]

    def _parse_gedcom_files(self):
        """Parse loaded GEDCOM files."""
        if not self.gedcom_files:
            messagebox.showwarning("Warning", "No GEDCOM files loaded")
            return

        def parse_task():
            try:
                import traceback

                all_individuals = {}
                all_families = {}

                for file in self.gedcom_files:
                    print(f"Parsing file: {file}")  # Debug
                    individuals, families = parse_gedcom_file(file)
                    all_individuals.update(individuals)
                    all_families.update(families)
                    print(
                        f"Parsed {len(individuals)} individuals, {len(families)} families from {file}"
                    )  # Debug

                return ("parse_complete", (all_individuals, all_families))
            except Exception as e:
                error_msg = f"Error parsing GEDCOM files: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                return ("error", error_msg)

        self._run_task(parse_task, "Parsing GEDCOM files...")

    def _train_model(self):
        """Train the OhanaAI model."""
        if not self.individuals:
            messagebox.showwarning(
                "Warning", "No data loaded. Please load GEDCOM files first."
            )
            return

        def train_task():
            try:
                import traceback

                trainer = OhanaAITrainer()
                trainer.prepare_data(self.individuals, self.families)
                training_history = trainer.train()
                return ("train_complete", (trainer, training_history))
            except Exception as e:
                error_msg = f"Error training model: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
                logger.error(error_msg)
                return ("error", error_msg)

        self._run_task(train_task, "Training model...")

    def _load_model(self):
        """Load a trained model."""
        model_file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.npz"), ("All files", "*.*")],
        )

        if model_file:

            def load_task():
                try:
                    predictor = OhanaAIPredictor()
                    predictor.load_model(model_file)
                    return ("model_loaded", predictor)
                except Exception as e:
                    return ("error", f"Error loading model: {str(e)}")

            self._run_task(load_task, "Loading model...")

    def _run_predictions(self):
        """Run parent predictions."""
        if not self.predictor:
            messagebox.showwarning(
                "Warning", "No model loaded. Please train or load a model first."
            )
            return

        if not self.individuals:
            messagebox.showwarning(
                "Warning", "No data loaded. Please load GEDCOM files first."
            )
            return

        def predict_task():
            try:
                self.predictor.prepare_data(self.individuals, self.families)
                predictions = self.predictor.predict_missing_parents()
                return ("predictions_complete", predictions)
            except Exception as e:
                return ("error", f"Error running predictions: {str(e)}")

        self._run_task(predict_task, "Running predictions...")

    def _detect_duplicates(self):
        """Detect duplicate individuals."""
        if not self.individuals:
            messagebox.showwarning(
                "Warning", "No data loaded. Please load GEDCOM files first."
            )
            return

        def detect_task():
            try:
                engine = DeduplicationEngine()
                for i, file in enumerate(self.gedcom_files):
                    individuals, families = parse_gedcom_file(file)
                    engine.add_gedcom_data(individuals, families, file)

                duplicates = engine.detect_duplicates()
                return ("duplicates_detected", (engine, duplicates))
            except Exception as e:
                return ("error", f"Error detecting duplicates: {str(e)}")

        self._run_task(detect_task, "Detecting duplicates...")

    def _review_duplicates(self):
        """Review detected duplicates."""
        if not self.duplicates:
            messagebox.showinfo(
                "Info", "No duplicates detected. Run duplicate detection first."
            )
            return

        self._populate_duplicates_table()

    def _run_task(self, task_func, status_message):
        """Run a task in background thread."""
        self.progress_var.set(status_message)
        self.progress_bar.start()
        self._update_status(status_message)

        thread = threading.Thread(target=lambda: self.result_queue.put(task_func()))
        thread.daemon = True
        thread.start()

    def _check_results(self):
        """Check for completed background tasks."""
        try:
            while True:
                result_type, result_data = self.result_queue.get_nowait()

                if result_type == "log":
                    self.log_text.insert(tk.END, result_data + "\n")
                    self.log_text.see(tk.END)

                elif result_type == "parse_complete":
                    self.individuals, self.families = result_data
                    self._update_status(
                        f"Parsed {len(self.individuals)} individuals, {len(self.families)} families"
                    )
                    self.progress_bar.stop()

                elif result_type == "train_complete":
                    self.trainer, history = result_data
                    self._update_status("Model training completed")
                    self.progress_bar.stop()
                    messagebox.showinfo(
                        "Success", "Model training completed successfully!"
                    )

                elif result_type == "model_loaded":
                    self.predictor = result_data
                    self._update_status("Model loaded successfully")
                    self.progress_bar.stop()
                    messagebox.showinfo("Success", "Model loaded successfully!")

                elif result_type == "predictions_complete":
                    self.predictions = result_data
                    self._populate_predictions_table()
                    self._update_status(
                        f"Generated {len(self.predictions)} predictions"
                    )
                    self.progress_bar.stop()

                elif result_type == "duplicates_detected":
                    self.dedup_engine, self.duplicates = result_data
                    self._populate_duplicates_table()
                    self._update_status(
                        f"Found {len(self.duplicates)} potential duplicates"
                    )
                    self.progress_bar.stop()

                elif result_type == "error":
                    messagebox.showerror("Error", result_data)
                    self.progress_bar.stop()
                    self._update_status("Error occurred")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._check_results)

    def _populate_predictions_table(self):
        """Populate the predictions table."""
        # Clear existing items
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)

        # Add predictions
        for pred in self.predictions:
            if pred.confidence_score >= self.filter_var.get():
                age_diff = (
                    f"{pred.age_difference} years" if pred.age_difference else "Unknown"
                )
                constraints = "✓" if pred.constraints_satisfied else "✗"

                self.pred_tree.insert(
                    "",
                    tk.END,
                    values=(
                        pred.child_name,
                        pred.candidate_parent_name,
                        f"{pred.confidence_score:.3f}",
                        age_diff,
                        constraints,
                    ),
                )

    def _populate_duplicates_table(self):
        """Populate the duplicates table."""
        # Clear existing items
        for item in self.dup_tree.get_children():
            self.dup_tree.delete(item)

        # Add duplicates
        for dup in self.duplicates:
            ind1_name = self.individuals.get(
                dup.individual1_id, type("", (), {"full_name": "Unknown"})
            ).full_name
            ind2_name = self.individuals.get(
                dup.individual2_id, type("", (), {"full_name": "Unknown"})
            ).full_name

            self.dup_tree.insert(
                "",
                tk.END,
                values=(
                    ind1_name,
                    ind2_name,
                    f"{dup.similarity_score:.3f}",
                    "; ".join(dup.reasons),
                ),
            )

    def _filter_predictions(self, value):
        """Filter predictions by confidence threshold."""
        self._populate_predictions_table()

    def _merge_selected_duplicate(self):
        """Merge selected duplicate."""
        selection = self.dup_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a duplicate to merge")
            return

        # Implementation would go here
        messagebox.showinfo("Info", "Merge functionality not yet implemented")

    def _ignore_duplicate(self):
        """Ignore selected duplicate."""
        selection = self.dup_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a duplicate to ignore")
            return

        # Remove from tree
        self.dup_tree.delete(selection[0])

    def _show_family_tree(self):
        """Show family tree visualization."""
        if not self.individuals:
            messagebox.showwarning("Warning", "No data loaded")
            return

        # Simple text-based family tree
        tree_text = "Family Tree Summary:\n\n"
        tree_text += f"Total Individuals: {len(self.individuals)}\n"
        tree_text += f"Total Families: {len(self.families)}\n\n"

        # Show some sample individuals
        tree_text += "Sample Individuals:\n"
        for i, (id, individual) in enumerate(list(self.individuals.items())[:10]):
            tree_text += f"- {individual.full_name} ({individual.birth_year or 'Unknown'}-{individual.death_year or 'Unknown'})\n"
            if i >= 9:
                tree_text += f"... and {len(self.individuals) - 10} more\n"
                break

        self.vis_text.delete(1.0, tk.END)
        self.vis_text.insert(1.0, tree_text)

    def _show_statistics(self):
        """Show data statistics."""
        if not self.individuals:
            messagebox.showwarning("Warning", "No data loaded")
            return

        stats_text = "Data Statistics:\n\n"

        # Gender distribution
        genders = {"M": 0, "F": 0, "U": 0}
        for ind in self.individuals.values():
            genders[ind.gender] = genders.get(ind.gender, 0) + 1

        stats_text += f"Gender Distribution:\n"
        stats_text += f"  Male: {genders['M']}\n"
        stats_text += f"  Female: {genders['F']}\n"
        stats_text += f"  Unknown: {genders['U']}\n\n"

        # Birth years
        birth_years = [
            ind.birth_year for ind in self.individuals.values() if ind.birth_year
        ]
        if birth_years:
            stats_text += f"Birth Years:\n"
            stats_text += f"  Earliest: {min(birth_years)}\n"
            stats_text += f"  Latest: {max(birth_years)}\n"
            stats_text += f"  Known: {len(birth_years)}/{len(self.individuals)}\n\n"

        # Missing parents
        missing_parents = 0
        for ind in self.individuals.values():
            if not ind.parent_families:
                missing_parents += 1

        stats_text += f"Missing Parents: {missing_parents}/{len(self.individuals)}\n"

        self.vis_text.delete(1.0, tk.END)
        self.vis_text.insert(1.0, stats_text)

    def _save_predictions(self):
        """Save predictions to file."""
        if not self.predictions:
            messagebox.showwarning("Warning", "No predictions to save")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if filename and self.predictor:
            self.predictor.export_predictions_csv(
                self.predictions, os.path.basename(filename)
            )
            messagebox.showinfo("Success", f"Predictions saved to {filename}")

    def _export_gedcom(self):
        """Export predictions as GEDCOM."""
        if not self.predictions:
            messagebox.showwarning("Warning", "No predictions to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export GEDCOM",
            defaultextension=".ged",
            filetypes=[("GEDCOM files", "*.ged"), ("All files", "*.*")],
        )

        if filename and self.predictor:
            self.predictor.export_predictions_gedcom(
                self.predictions, os.path.basename(filename)
            )
            messagebox.showinfo("Success", f"GEDCOM exported to {filename}")

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About OhanaAI",
            "OhanaAI - Genealogical Parent Prediction\n\n"
            "Uses Graph Neural Networks to predict missing parents\n"
            "in genealogical family trees.\n\n"
            "Built with MLX and Tkinter\n"
            "Version 1.0",
        )

    def _update_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()

    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for GUI."""
    app = OhanaAIGUI()
    app.run()


if __name__ == "__main__":
    main()
