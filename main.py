import os
import sys
import subprocess
import signal

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QEvent
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QDialog,
    QPlainTextEdit,
    QHBoxLayout,
    QLabel,
)

DLO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGX_SETUP_ENV = "/opt/Algoryx/AGX-2.40.1.5/setup_env.bash"

DATASET_GENERATOR_PATH = os.path.join(
    DLO_ROOT, "agxLibrary", "dataset_generator_launch.py"
)
TRAINING_LAUNCHER_PATH = os.path.join(DLO_ROOT, "dlo_diffusion", "train.py")
PREDICTION_LAUNCHER_PATH = os.path.join(DLO_ROOT, "dlo_diffusion", "predict.py")


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


class ProcessWindow(QDialog):
    def __init__(self, *, title: str, program: str, args: list[str], cwd: str):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(900, 550)

        self._killed = False

        self.proc = QProcess(self)
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(cwd)

        env = QProcessEnvironment.systemEnvironment()
        # Ensure the process sees any env changes we applied in Python
        for k, v in os.environ.items():
            env.insert(k, v)
        self.proc.setProcessEnvironment(env)

        self.status_lbl = QLabel("Not started")
        self.status_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_process)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        top = QHBoxLayout()
        top.addWidget(self.status_lbl, 1)
        top.addWidget(self.stop_btn)
        top.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.log)

        self.proc.started.connect(self._on_started)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)

        # Unbuffered Python output makes logs show up immediately.
        self.proc.start()

    def _append(self, text: str) -> None:
        self.log.appendPlainText(text.rstrip("\n"))

    def _on_started(self) -> None:
        self.status_lbl.setText(f"Running (pid={self.proc.processId()})")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        st = "Crash" if exit_status == QProcess.CrashExit else "Exit"
        killed = " (killed)" if self._killed else ""
        self.status_lbl.setText(f"{st} code={exit_code}{killed}")
        self.stop_btn.setEnabled(False)

    def _on_error(self, err: QProcess.ProcessError) -> None:
        self._append(f"[QProcess error] {err}")
        # If it failed to start, the process is not running.
        if self.proc.state() == QProcess.NotRunning:
            self.stop_btn.setEnabled(False)

    def _on_stdout(self) -> None:
        data = bytes(self.proc.readAllStandardOutput())
        if data:
            self._append(data.decode("utf-8", "replace"))

    def _on_stderr(self) -> None:
        data = bytes(self.proc.readAllStandardError())
        if data:
            self._append(data.decode("utf-8", "replace"))

    def stop_process(self) -> None:
        if self.proc.state() == QProcess.NotRunning:
            return

        self._killed = True
        self.status_lbl.setText("Stopping...")

        self.proc.kill()
        # Force-kill after a short grace period
        QTimer.singleShot(1000, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def closeEvent(self, event: QEvent) -> None:
        # Keep behavior simple: closing the window does not auto-kill.
        # If you want close=kill, call self.stop_process() here.
        super().closeEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Launcher")

        self._process_windows: list[ProcessWindow] = []

        layout = QVBoxLayout(self)

        self.dataset_generator_btn = QPushButton("Generate Dataset")
        self.training_btn = QPushButton("Train Model")
        self.prediction_btn = QPushButton("Run Prediction")

        self.dataset_generator_btn.clicked.connect(self.run_dataset_generator)
        self.training_btn.clicked.connect(self.run_training)
        self.prediction_btn.clicked.connect(self.run_prediction)

        layout.addWidget(self.dataset_generator_btn)
        layout.addWidget(self.training_btn)
        layout.addWidget(self.prediction_btn)

    def _spawn_logged_process(self, script_path: str, kind: str) -> None:
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "Missing script", f"Not found:\n{script_path}")
            return

        program = sys.executable
        args = ["-u", script_path]  # -u = unbuffered stdout/stderr
        cwd = os.path.dirname(script_path)

        w = ProcessWindow(title=f"{kind} log", program=program, args=args, cwd=cwd)
        w.setAttribute(Qt.WA_DeleteOnClose, True)
        w.destroyed.connect(
            lambda *_: (
                self._process_windows.remove(w) if w in self._process_windows else None
            )
        )
        self._process_windows.append(w)
        w.show()

    def run_training(self):
        self._spawn_logged_process(TRAINING_LAUNCHER_PATH, "Training")

    def run_prediction(self):
        self._spawn_logged_process(PREDICTION_LAUNCHER_PATH, "Prediction")

    def run_dataset_generator(self):
        self._spawn_logged_process(DATASET_GENERATOR_PATH, "Dataset generator")


def main():
    app = QApplication(sys.argv)

    # Close the GUI on Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    w = MainWindow()
    w.resize(320, 120)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
