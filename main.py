import os
import sys
import subprocess
import signal
import shutil
import json
import yaml

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QMessageBox,
    QDialog,
    QLabel,
    QSpinBox,
    QCheckBox,
    QDialogButtonBox,
    QGroupBox,
    QDoubleSpinBox,
    QLineEdit,
    QScrollArea,
    QTabWidget,
    QComboBox,
)

DLO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGX_SETUP_ENV = "/opt/Algoryx/AGX-2.40.1.5/setup_env.bash"
CONFIG_CACHE_FILE = os.path.join(DLO_ROOT, "gui", ".launcher_cache.json")

DATASET_GENERATOR_PATH = os.path.join(
    DLO_ROOT, "agxLibrary", "dataset_generator_launch.py"
)
TRAINING_LAUNCHER_PATH = os.path.join(DLO_ROOT, "dlo_diffusion", "train.py")
PREDICTION_LAUNCHER_PATH = os.path.join(DLO_ROOT, "dlo_diffusion", "predict.py")

# Config file paths
DATASET_CONFIG_PATH = os.path.join(
    DLO_ROOT, "agxLibrary", "agxLibrary", "config", "dataset_generator_config.yaml"
)
TRAINING_CONFIG_PATH = os.path.join(
    DLO_ROOT, "dlo_diffusion", "dlo_diffusion", "config", "trainer.yaml"
)
PREDICTION_CONFIG_PATH = os.path.join(
    DLO_ROOT, "dlo_diffusion", "dlo_diffusion", "config", "predict.yaml"
)


def _update_os_environ_from_bash_source(bash_script: str) -> None:
    if not os.path.exists(bash_script):
        return

    cmd = f"source '{bash_script}' >/dev/null 2>&1 && env -0"
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for item in proc.stdout.split(b"\x00"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        os.environ[k.decode("utf-8", "ignore")] = v.decode("utf-8", "ignore")


try:
    _update_os_environ_from_bash_source(AGX_SETUP_ENV)
except Exception:
    pass

os.environ["PYTHONPATH"] = f"{DLO_ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")


def _find_terminal() -> tuple[str, list[str]]:
    """Find available terminal emulator and return command template."""
    terminals = [
        ("gnome-terminal", ["gnome-terminal", "--", "bash", "-c"]),
        ("konsole", ["konsole", "-e", "bash", "-c"]),
        ("xfce4-terminal", ["xfce4-terminal", "-e", "bash", "-c"]),
        ("xterm", ["xterm", "-e", "bash", "-c"]),
        ("alacritty", ["alacritty", "-e", "bash", "-c"]),
        ("kitty", ["kitty", "bash", "-c"]),
        ("terminator", ["terminator", "-e", "bash", "-c"]),
    ]

    for name, cmd in terminals:
        if shutil.which(name):
            return name, cmd

    return None, None


def load_cached_config(key: str) -> dict:
    """Load cached configuration for a specific launcher."""
    if os.path.exists(CONFIG_CACHE_FILE):
        try:
            with open(CONFIG_CACHE_FILE, "r") as f:
                cache = json.load(f)
                return cache.get(key, {})
        except Exception:
            pass
    return {}


def save_cached_config(key: str, config: dict) -> None:
    """Save configuration to cache."""
    cache = {}
    if os.path.exists(CONFIG_CACHE_FILE):
        try:
            with open(CONFIG_CACHE_FILE, "r") as f:
                cache = json.load(f)
        except Exception:
            pass

    cache[key] = config

    try:
        with open(CONFIG_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Failed to save config cache: {e}")


class ConfigEditorDialog(QDialog):
    """Generic dialog for editing YAML configurations."""

    def __init__(self, title: str, config_path: str, cache_key: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 600)
        self.config_path = config_path
        self.cache_key = cache_key

        # Load original config
        self.original_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.original_config = yaml.safe_load(f)

        # Load cached overrides
        self.cached_overrides = load_cached_config(cache_key)

        # Main layout
        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            f"Edit configuration parameters. Changes override: {os.path.basename(config_path)}"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Scroll area for config fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.fields_layout = QVBoxLayout(scroll_content)

        # Build config UI
        self.fields = {}
        self._build_config_ui(self.original_config, self.cached_overrides, [])

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

        layout.addLayout(button_layout)

    def _build_config_ui(self, config: dict, overrides: dict, path: list) -> None:
        """Recursively build UI fields for config."""
        for key, value in config.items():
            current_path = path + [key]
            path_str = ".".join(current_path)

            # Skip certain keys
            if key in ["defaults", "_self_"]:
                continue

            if isinstance(value, dict):
                # Add a group header
                header = QLabel(f"<b>{' > '.join(current_path)}</b>")
                header.setStyleSheet("margin-top: 15px; margin-bottom: 5px;")
                self.fields_layout.addWidget(header)

                # Recursively handle nested dict
                self._build_config_ui(value, overrides, current_path)

            elif isinstance(value, list):
                # Handle lists (show as comma-separated string)
                row_layout = QHBoxLayout()
                label = QLabel(f"{key}:")
                label.setMinimumWidth(200)
                row_layout.addWidget(label)

                line_edit = QLineEdit()
                cached_value = self._get_nested_value(overrides, current_path)
                if cached_value is not None:
                    line_edit.setText(str(cached_value))
                else:
                    line_edit.setText(str(value))
                line_edit.setPlaceholderText(f"Default: {value}")
                row_layout.addWidget(line_edit)

                self.fields_layout.addLayout(row_layout)
                self.fields[path_str] = ("list", line_edit, value)

            elif isinstance(value, bool):
                # Boolean field
                row_layout = QHBoxLayout()
                label = QLabel(f"{key}:")
                label.setMinimumWidth(200)
                row_layout.addWidget(label)

                checkbox = QCheckBox()
                cached_value = self._get_nested_value(overrides, current_path)
                if cached_value is not None:
                    checkbox.setChecked(bool(cached_value))
                else:
                    checkbox.setChecked(value)
                row_layout.addWidget(checkbox)
                row_layout.addStretch()

                self.fields_layout.addLayout(row_layout)
                self.fields[path_str] = ("bool", checkbox, value)

            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Numeric field
                row_layout = QHBoxLayout()
                label = QLabel(f"{key}:")
                label.setMinimumWidth(200)
                row_layout.addWidget(label)

                if isinstance(value, float):
                    spinbox = QDoubleSpinBox()
                    spinbox.setDecimals(6)
                    spinbox.setRange(-1e10, 1e10)
                    spinbox.setSingleStep(0.001 if abs(value) < 1 else 1.0)
                else:
                    spinbox = QSpinBox()
                    spinbox.setRange(-2147483648, 2147483647)

                cached_value = self._get_nested_value(overrides, current_path)
                if cached_value is not None:
                    spinbox.setValue(
                        float(cached_value)
                        if isinstance(value, float)
                        else int(cached_value)
                    )
                else:
                    spinbox.setValue(value)

                row_layout.addWidget(spinbox)

                self.fields_layout.addLayout(row_layout)
                self.fields[path_str] = (
                    "float" if isinstance(value, float) else "int",
                    spinbox,
                    value,
                )

            elif isinstance(value, str):
                # String field
                row_layout = QHBoxLayout()
                label = QLabel(f"{key}:")
                label.setMinimumWidth(200)
                row_layout.addWidget(label)

                line_edit = QLineEdit()
                cached_value = self._get_nested_value(overrides, current_path)
                if cached_value is not None:
                    line_edit.setText(str(cached_value))
                else:
                    line_edit.setText(value)
                line_edit.setPlaceholderText(f"Default: {value}")
                row_layout.addWidget(line_edit)

                self.fields_layout.addLayout(row_layout)
                self.fields[path_str] = ("str", line_edit, value)

    def _get_nested_value(self, d: dict, path: list):
        """Get value from nested dict using path."""
        current = d
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _set_nested_value(self, d: dict, path: list, value) -> None:
        """Set value in nested dict using path."""
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def reset_to_defaults(self) -> None:
        """Reset all fields to default values."""
        for path_str, (field_type, widget, default_value) in self.fields.items():
            if field_type == "bool":
                widget.setChecked(default_value)
            elif field_type in ["int", "float"]:
                widget.setValue(default_value)
            elif field_type in ["str", "list"]:
                widget.setText(str(default_value))

    def get_overrides(self) -> dict:
        """Get all override values as a nested dict."""
        overrides = {}

        for path_str, (field_type, widget, default_value) in self.fields.items():
            path = path_str.split(".")

            if field_type == "bool":
                value = widget.isChecked()
            elif field_type == "int":
                value = widget.value()
            elif field_type == "float":
                value = widget.value()
            elif field_type == "str":
                value = widget.text()
            elif field_type == "list":
                # Parse list from string
                text = widget.text().strip()
                try:
                    value = yaml.safe_load(f"[{text}]") if text else default_value
                except:
                    value = text

            # Only include if different from default
            if value != default_value:
                self._set_nested_value(overrides, path, value)

        return overrides

    def accept(self) -> None:
        """Save configuration and close."""
        overrides = self.get_overrides()
        save_cached_config(self.cache_key, overrides)
        super().accept()


class DatasetGeneratorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Generator Configuration")
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        # Tabs
        tabs = QTabWidget()

        # Tab 1: Launch parameters
        launch_tab = QWidget()
        launch_layout = QVBoxLayout(launch_tab)

        parallel_group = QGroupBox("Parallel Execution")
        parallel_layout = QGridLayout()

        parallel_layout.addWidget(QLabel("Number of parallel simulations (-n):"), 0, 0)
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setMinimum(1)
        self.n_spinbox.setMaximum(128)
        self.n_spinbox.setValue(1)
        parallel_layout.addWidget(self.n_spinbox, 0, 1)

        self.all_cpus_checkbox = QCheckBox("Use all CPU cores (--all_cpus)")
        parallel_layout.addWidget(self.all_cpus_checkbox, 1, 0, 1, 2)

        parallel_group.setLayout(parallel_layout)
        launch_layout.addWidget(parallel_group)

        execution_group = QGroupBox("Execution Count")
        execution_layout = QGridLayout()

        execution_layout.addWidget(QLabel("Number of executions (-e):"), 0, 0)
        self.e_spinbox = QSpinBox()
        self.e_spinbox.setMinimum(1)
        self.e_spinbox.setMaximum(10000)
        self.e_spinbox.setValue(1)
        execution_layout.addWidget(self.e_spinbox, 0, 1)

        execution_layout.addWidget(QLabel("OR Total samples (--total_samples):"), 1, 0)
        self.total_samples_spinbox = QSpinBox()
        self.total_samples_spinbox.setMinimum(0)
        self.total_samples_spinbox.setMaximum(1000000)
        self.total_samples_spinbox.setValue(0)
        self.total_samples_spinbox.setSpecialValueText("Not set")
        execution_layout.addWidget(self.total_samples_spinbox, 1, 1)

        execution_group.setLayout(execution_layout)
        launch_layout.addWidget(execution_group)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.a_checkbox = QCheckBox("Run with -a flag (AGX viewer automation)")
        options_layout.addWidget(self.a_checkbox)

        options_group.setLayout(options_layout)
        launch_layout.addWidget(options_group)

        launch_layout.addStretch()
        tabs.addTab(launch_tab, "Launch Parameters")

        # Tab 2: Config editor button
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)

        config_info = QLabel(
            "Click below to edit dataset generation configuration parameters"
        )
        config_info.setWordWrap(True)
        config_layout.addWidget(config_info)

        edit_config_btn = QPushButton("📝 Edit Configuration File")
        edit_config_btn.clicked.connect(self.edit_config)
        config_layout.addWidget(edit_config_btn)

        config_layout.addStretch()
        tabs.addTab(config_tab, "Configuration")

        layout.addWidget(tabs)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load cached launch params
        cached = load_cached_config("dataset_launch")
        if cached:
            self.n_spinbox.setValue(cached.get("n", 1))
            self.e_spinbox.setValue(cached.get("e", 1))
            self.total_samples_spinbox.setValue(cached.get("total_samples", 0))
            self.all_cpus_checkbox.setChecked(cached.get("all_cpus", False))
            self.a_checkbox.setChecked(cached.get("a", False))

    def edit_config(self) -> None:
        """Open config editor dialog."""
        dialog = ConfigEditorDialog(
            "Dataset Generator Configuration",
            DATASET_CONFIG_PATH,
            "dataset_config",
            self,
        )
        dialog.exec()

    def get_parameters(self) -> dict:
        """Return the selected parameters as a dictionary."""
        params = {}

        if self.all_cpus_checkbox.isChecked():
            params["--all_cpus"] = True
        else:
            params["-n"] = self.n_spinbox.value()

        if self.total_samples_spinbox.value() > 0:
            params["--total_samples"] = self.total_samples_spinbox.value()
        else:
            params["-e"] = self.e_spinbox.value()

        if self.a_checkbox.isChecked():
            params["-a"] = True

        return params

    def accept(self) -> None:
        """Save launch parameters and close."""
        launch_params = {
            "n": self.n_spinbox.value(),
            "e": self.e_spinbox.value(),
            "total_samples": self.total_samples_spinbox.value(),
            "all_cpus": self.all_cpus_checkbox.isChecked(),
            "a": self.a_checkbox.isChecked(),
        }
        save_cached_config("dataset_launch", launch_params)
        super().accept()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLO Pipeline Launcher")
        self.setMinimumSize(400, 200)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("DLO Pipeline Launcher")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        self.dataset_generator_btn = QPushButton("📊 Generate Dataset")
        self.training_btn = QPushButton("🎓 Train Model")
        self.prediction_btn = QPushButton("🔮 Run Prediction")

        button_style = """
            QPushButton {
                padding: 15px;
                font-size: 14px;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """

        self.dataset_generator_btn.setStyleSheet(button_style)
        self.training_btn.setStyleSheet(
            button_style.replace("#4CAF50", "#2196F3")
            .replace("#45a049", "#1976D2")
            .replace("#3d8b40", "#1565C0")
        )
        self.prediction_btn.setStyleSheet(
            button_style.replace("#4CAF50", "#FF9800")
            .replace("#45a049", "#F57C00")
            .replace("#3d8b40", "#E65100")
        )

        self.dataset_generator_btn.clicked.connect(self.run_dataset_generator)
        self.training_btn.clicked.connect(self.run_training)
        self.prediction_btn.clicked.connect(self.run_prediction)

        main_layout.addWidget(self.dataset_generator_btn)
        main_layout.addWidget(self.training_btn)
        main_layout.addWidget(self.prediction_btn)

        main_layout.addStretch()

    def _build_hydra_overrides(self, overrides: dict, prefix: str = "") -> list:
        """Convert nested dict to Hydra CLI override format."""
        result = []
        for key, value in overrides.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.extend(self._build_hydra_overrides(value, full_key))
            else:
                result.append(f"{full_key}={value}")
        return result

    def _run_script_in_new_terminal(
        self, script_path: str, kind: str, extra_args: list = None
    ) -> None:
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "Missing script", f"Not found:\n{script_path}")
            return

        terminal_name, terminal_cmd = _find_terminal()

        if not terminal_cmd:
            QMessageBox.critical(
                self,
                "No Terminal Found",
                "Could not find a terminal emulator. Please install one of:\n"
                "gnome-terminal, konsole, xfce4-terminal, xterm, alacritty, kitty, or terminator",
            )
            return

        bash_commands = []

        conda_base = (
            os.environ.get("CONDA_EXE", "")
            .replace("/bin/conda", "")
            .replace("/condabin/conda", "")
        )
        if not conda_base:
            conda_base = os.path.expanduser("~/miniconda3")
            if not os.path.exists(conda_base):
                conda_base = os.path.expanduser("~/anaconda3")

        conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        if os.path.exists(conda_sh):
            bash_commands.append(f"source '{conda_sh}'")

        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "pyel_agx")
        bash_commands.append(f"conda activate {conda_env}")

        if os.path.exists(AGX_SETUP_ENV):
            bash_commands.append(f"source '{AGX_SETUP_ENV}'")

        pythonpath = f"{DLO_ROOT}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
        bash_commands.append(f"export PYTHONPATH='{pythonpath}'")

        bash_commands.append(f"cd '{os.path.dirname(script_path)}'")

        bash_commands.append(f"echo '{'='*60}'")
        bash_commands.append(f"echo 'Running: {kind}'")
        bash_commands.append(f"echo 'Script: {script_path}'")
        if extra_args:
            bash_commands.append(f"echo 'Arguments: {' '.join(extra_args)}'")
        bash_commands.append(f"echo '{'='*60}'")
        bash_commands.append("echo")

        cmd_parts = ["python", "-u", f"'{script_path}'"]
        if extra_args:
            cmd_parts.extend(extra_args)
        bash_commands.append(" ".join(cmd_parts))

        bash_commands.append("echo")
        bash_commands.append(
            "echo 'Process finished. Press Enter to close this terminal...'"
        )
        bash_commands.append("read")

        full_command = "; ".join(bash_commands)

        cmd = terminal_cmd + [full_command]

        env = os.environ.copy()
        vars_to_clean = ["LD_LIBRARY_PATH", "LD_PRELOAD"]
        for var in vars_to_clean:
            if var in env:
                del env[var]

        subprocess.Popen(cmd, env=env, start_new_session=True)

    def run_training(self):
        dialog = ConfigEditorDialog(
            "Training Configuration", TRAINING_CONFIG_PATH, "training_config", self
        )
        if dialog.exec() == QDialog.Accepted:
            overrides = dialog.get_overrides()
            args = self._build_hydra_overrides(overrides)
            self._run_script_in_new_terminal(TRAINING_LAUNCHER_PATH, "Training", args)

    def run_prediction(self):
        dialog = ConfigEditorDialog(
            "Prediction Configuration",
            PREDICTION_CONFIG_PATH,
            "prediction_config",
            self,
        )
        if dialog.exec() == QDialog.Accepted:
            overrides = dialog.get_overrides()
            args = self._build_hydra_overrides(overrides)
            self._run_script_in_new_terminal(
                PREDICTION_LAUNCHER_PATH, "Prediction", args
            )

    def run_dataset_generator(self):
        dialog = DatasetGeneratorDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_parameters()

            args = []
            for key, value in params.items():
                if isinstance(value, bool):
                    args.append(key)
                else:
                    args.append(key)
                    args.append(str(value))

            self._run_script_in_new_terminal(
                DATASET_GENERATOR_PATH, "Dataset Generator", args
            )


def main():
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    w = MainWindow()
    w.resize(500, 400)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
