import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QScrollArea, QGroupBox, QGridLayout, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ---------------------------
# Subplot 配置的 Widget
# ---------------------------
class SubplotConfigWidget(QWidget):
    def __init__(self, data_items, subplot_index, parent=None):
        """
        data_items: List of tuples (file_index, column_name, file_basename)
        subplot_index: 此 subplot 的索引
        """
        super().__init__(parent)
        self.subplot_index = subplot_index
        self.data_items = data_items  # 所有可供選取的資料項目
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel(f"Subplot {self.subplot_index+1}")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)
        
        # 使用 QListWidget 列出所有「檔名 - 欄位名稱」的選項，讓使用者選擇要顯示的資料
        self.dataListWidget = QListWidget()
        for item in self.data_items:
            file_idx, col_name, file_base = item
            list_item = QListWidgetItem(f"{file_base} - {col_name}")
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Unchecked)
            # 將 (file_index, col_name) 存到 item 的 UserRole 中以便後續取回
            list_item.setData(Qt.UserRole, (file_idx, col_name))
            self.dataListWidget.addItem(list_item)
        layout.addWidget(QLabel("選擇資料 (勾選):"))
        layout.addWidget(self.dataListWidget)
        
        # 低通濾波設定
        flayout = QHBoxLayout()
        flayout.addWidget(QLabel("低通截止頻率 (Hz, 0=不套用):"))
        self.lpCutoffSpin = QDoubleSpinBox()
        self.lpCutoffSpin.setRange(0, 10000)
        self.lpCutoffSpin.setSingleStep(1)
        self.lpCutoffSpin.setValue(0)
        flayout.addWidget(self.lpCutoffSpin)
        layout.addLayout(flayout)
        
        # FFT 轉換選項
        self.fftCheckbox = QCheckBox("轉成頻域 FFT")
        layout.addWidget(self.fftCheckbox)
        
        # 新增偏移量設定
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("資料偏移量:"))
        self.offsetSpin = QDoubleSpinBox()
        self.offsetSpin.setRange(-100000, 100000)
        self.offsetSpin.setSingleStep(1)
        self.offsetSpin.setValue(0)
        offset_layout.addWidget(self.offsetSpin)
        layout.addLayout(offset_layout)
        
        self.setLayout(layout)
        
    def get_selected_data_items(self):
        """回傳這個 subplot 中被勾選之資料項目列表，每個項目為 (file_index, column_name)"""
        selected = []
        for i in range(self.dataListWidget.count()):
            item = self.dataListWidget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected
    
    def get_lp_cutoff(self):
        return self.lpCutoffSpin.value()
    
    def use_fft(self):
        return self.fftCheckbox.isChecked()
    
    def get_offset(self):
        return self.offsetSpin.value()


# ---------------------------
# Matplotlib Canvas
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


# ---------------------------
# 主視窗
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV 多檔案繪圖工具")
        # 儲存 CSV 檔案路徑、設定與資料
        self.csv_file_paths = []    # 每一筆是 CSV 檔路徑
        self.csv_data = []          # 每一筆為 pd.DataFrame（含 header)
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 左側：控制面板
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)
        control_widget.setMinimumWidth(350)
        main_layout.addWidget(control_widget)
        
        # 載入 CSV 檔案按鈕
        load_btn = QPushButton("載入 CSV 檔案")
        load_btn.clicked.connect(self.load_csv_files)
        control_layout.addWidget(load_btn)
        
        # 顯示已載入檔案與設定取樣頻率
        self.fileTable = QTableWidget(0, 2)
        self.fileTable.setHorizontalHeaderLabels(["檔案路徑", "取樣頻率 (Hz)"])
        self.fileTable.horizontalHeader().setStretchLastSection(True)
        control_layout.addWidget(QLabel("已載入檔案:"))
        control_layout.addWidget(self.fileTable)
        
        # Subplot 設定區
        sub_conf_group = QGroupBox("Subplot 設定")
        sub_conf_layout = QVBoxLayout()
        sub_conf_group.setLayout(sub_conf_layout)
        control_layout.addWidget(sub_conf_group)
        
        # 全局排列：列數與欄數
        grid_conf_layout = QHBoxLayout()
        grid_conf_layout.addWidget(QLabel("列數:"))
        self.rowsSpin = QSpinBox()
        self.rowsSpin.setRange(1, 10)
        self.rowsSpin.setValue(1)
        grid_conf_layout.addWidget(self.rowsSpin)
        grid_conf_layout.addWidget(QLabel("欄數:"))
        self.colsSpin = QSpinBox()
        self.colsSpin.setRange(1, 10)
        self.colsSpin.setValue(1)
        grid_conf_layout.addWidget(self.colsSpin)
        # 當排列設定變化時，更新 subplot 設定區
        self.rowsSpin.valueChanged.connect(self.update_subplot_config_widgets)
        self.colsSpin.valueChanged.connect(self.update_subplot_config_widgets)
        sub_conf_layout.addLayout(grid_conf_layout)
        
        # 使用捲軸區呈現每個 subplot 的詳細設定
        self.subplotConfigArea = QScrollArea()
        self.subplotConfigArea.setWidgetResizable(True)
        self.subplotConfigContainer = QWidget()
        self.subplotConfigLayout = QVBoxLayout()
        self.subplotConfigContainer.setLayout(self.subplotConfigLayout)
        self.subplotConfigArea.setWidget(self.subplotConfigContainer)
        sub_conf_layout.addWidget(self.subplotConfigArea)
        
        self.subplotConfigWidgets = []  # 儲存各 subplot 的設定 widget
        
        # 繪圖按鈕
        plot_btn = QPushButton("繪圖")
        plot_btn.clicked.connect(self.do_plot)
        control_layout.addWidget(plot_btn)
        
        # 右側：Matplotlib 繪圖區
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        main_layout.addWidget(plot_widget, 1)
        
        self.show()
    
    def load_csv_files(self):
        """利用檔案對話框選取一個或多個 CSV 檔案，並利用 pandas 讀取檔案（包含 header)"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "選擇 CSV 檔案", "", "CSV Files (*.csv)")
        if not file_paths:
            return
        # 清空先前資料
        self.csv_file_paths.clear()
        self.csv_data.clear()
        self.fileTable.setRowCount(0)
        
        for path in file_paths:
            try:
                # 讀取 CSV，第一列為 header
                df = pd.read_csv(path, header=0)
            except Exception as e:
                print(f"無法讀取 {path} : {e}")
                continue
            self.csv_file_paths.append(path)
            self.csv_data.append(df)
            row = self.fileTable.rowCount()
            self.fileTable.insertRow(row)
            self.fileTable.setItem(row, 0, QTableWidgetItem(path))
            # 預設取樣頻率為 1.0 Hz（使用者可於表格中修改）
            fs_item = QTableWidgetItem("1.0")
            self.fileTable.setItem(row, 1, fs_item)
            
        # 載入檔案後更新每個 subplot 的資料選項
        self.update_subplot_config_widgets()
    
    def update_subplot_config_widgets(self):
        """根據最新的檔案與排列設定，更新每個 subplot 的配置 widget
           資料項目列表格式：[(file_index, column_name, file_basename), ...]
        """
        rows = self.rowsSpin.value()
        cols = self.colsSpin.value()
        num_subplots = rows * cols
        
        # 清除先前 subplot widget
        for w in self.subplotConfigWidgets:
            w.setParent(None)
        self.subplotConfigWidgets = []
        
        # 產生可供選取的資料項目列表：遍歷每個 CSV 檔案的所有欄位
        data_items = []
        for file_idx, df in enumerate(self.csv_data):
            file_base = os.path.basename(self.csv_file_paths[file_idx])
            for col in df.columns:
                data_items.append((file_idx, col, file_base))
                
        # 為每個 subplot 建立個別設定 widget
        for i in range(num_subplots):
            widget = SubplotConfigWidget(data_items, subplot_index=i)
            self.subplotConfigWidgets.append(widget)
            self.subplotConfigLayout.addWidget(widget)
    
    def do_plot(self):
        """根據使用者設定，進行各 subplot 的繪圖處理：
           - 根據檔案取樣頻率產生時間軸（所有資料 t=0 對齊）
           - 若設定截止頻率則進行 Butterworth 低通濾波
           - 若勾選 FFT 則進行 FFT 計算並轉換至頻域顯示
        """
        # 取得各檔案的取樣頻率（若使用者修改 table 值，以 table 為準）
        file_fs = []
        for row in range(self.fileTable.rowCount()):
            item = self.fileTable.item(row, 1)
            try:
                fs = float(item.text())
            except:
                fs = 1.0
            file_fs.append(fs)
        
        # 清除舊圖，設定子圖排列
        self.canvas.fig.clear()
        rows = self.rowsSpin.value()
        cols = self.colsSpin.value()
        axes = self.canvas.fig.subplots(rows, cols)
        if rows * cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()
        
        # 逐一處理每個 subplot
        for idx, config_widget in enumerate(self.subplotConfigWidgets):
            if idx >= len(axes):
                break
            ax = axes[idx]
            ax.clear()
            # 取得使用者勾選的資料項目
            selected_items = config_widget.get_selected_data_items()  # 每個項目為 (file_index, column_name)
            lp_cutoff = config_widget.get_lp_cutoff()
            use_fft = config_widget.use_fft()
            offset = config_widget.get_offset()
            
            if not selected_items:
                ax.set_title(f"Subplot {idx+1} (無資料)")
                continue
            
            # 對每個選取的資料進行處理與繪圖
            for item in selected_items:
                file_idx, col_name = item
                if file_idx >= len(self.csv_data):
                    continue
                df = self.csv_data[file_idx]
                # 取出欄位資料並轉為 numpy array
                try:
                    data = df[col_name].values.astype(float)
                except Exception as e:
                    print(f"資料轉型失敗: {e}")
                    continue
                fs = file_fs[file_idx]
                t = np.arange(len(data)) / fs  # 每個檔案的時間軸，皆從 0 起始
                
                # 若設定截止頻率且大於 0 時，進行 Butterworth 低通濾波
                if lp_cutoff > 0:
                    nyq = 0.5 * fs
                    normal_cutoff = lp_cutoff / nyq
                    if normal_cutoff < 1:
                        try:
                            b, a = butter(4, normal_cutoff, btype='low', analog=False)
                            data_proc = filtfilt(b, a, data)
                        except Exception as e:
                            print(f"濾波失敗: {e}")
                            data_proc = data
                    else:
                        data_proc = data
                else:
                    data_proc = data
                    
                # 若勾選 FFT 模式，則計算 FFT 並換為頻域資料
                if use_fft:
                    N = len(data_proc)
                    fft_vals = np.fft.rfft(data_proc)
                    fft_freq = np.fft.rfftfreq(N, d=1/fs)
                    x_vals = fft_freq
                    y_vals = np.abs(fft_vals)
                    xlabel = "Frequency (Hz)"
                else:
                    x_vals = t
                    y_vals = data_proc + offset
                    xlabel = "Time (t)"
                    
                # 使用「檔名 - 欄位名稱」作為圖例標籤
                file_base = os.path.basename(self.csv_file_paths[file_idx])
                label_text = f"{file_base} - {col_name}"
                ax.plot(x_vals, y_vals, label=label_text)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            if len(selected_items) > 1:
                ax.legend()
            ax.set_title(f"Subplot {idx+1}")
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
