import sys
from PyQt5.QtCore import pyqtSignal, QRect, QPoint, Qt, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QSpinBox, QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QImage, QColor, QBrush, QPen


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.central_widget = CentralWidget(self)
        self.setCentralWidget(self.central_widget)
        self.show()


class CentralWidget(QWidget):
    def __init__(self, parent):
        super(CentralWidget, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.annotator_config = AnnotatorConfigurationModel()

        splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget(self)
        self.config_editor = ConfigurationEditor(self, self.annotator_config)
        self.tabs.addTab(self.config_editor, "Configuration")
        splitter.addWidget(self.tabs)

        self.canvas = ImageCanvas(self, self.annotator_config)
        splitter.addWidget(self.canvas)

        # TODO
        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Maximum)
        policy.setVerticalPolicy(QSizePolicy.Preferred)
        self.canvas.setSizePolicy(policy)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.layout.addWidget(splitter)


class ConfigurationEditor(QWidget):
    def __init__(self, parent, model):
        super(ConfigurationEditor, self).__init__(parent)
        self.model = model
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.color_selector_description = QLabel(self)
        self.color_selector_description.setText("Select color:")
        self.layout.addWidget(self.color_selector_description)
        self.color_selector = QSpinBox(self)
        # yes, we start counting at 0
        self.color_selector.setRange(0, model.n_classes - 1)
        self.color_selector.valueChanged.connect(self.change_active_color)
        self.layout.addWidget(self.color_selector)

    def change_active_color(self, color):
        self.model.active_color = color


class ImageCanvas(QWidget):
    def __init__(self, parent, config):
        super(ImageCanvas, self).__init__(parent)
        self.config = config

        self.original_img = QImage("image.jpg")
        self.img_view = ImageView(self)
        self.initUI(self.original_img)
        self.img_view.start_drag.connect(self.start_drag)
        self.img_view.drag.connect(self.drag)
        self.img_view.stop_drag.connect(self.stop_drag)

        # temporary variables
        self.started_drag = None

        # results
        self.bounding_boxes = []

    def initUI(self, img):
        self.setWindowTitle("Image annotator")
        #self.setGeometry(10, 10, 640, 480)
    
        overlay = self._new_overlay(img)
        self._apply_and_show_overlay(img, overlay)

    def start_drag(self, x, y):
        self.started_drag = self._apply_bounds(x, y)

    def drag(self, x, y):
        overlay = self._new_overlay(self.original_img)
        x, y = self._apply_bounds(x, y)

        # TODO could be done directly, without overlay image
        painter = QPainter()
        painter.begin(overlay)
        for topleft, bottomright, color in self.bounding_boxes:
            self._draw_rect(painter, topleft, bottomright, color)
        self._draw_rect(painter, self.started_drag, (x, y), self.config.active_color)
        painter.end()

        self._apply_and_show_overlay(self.original_img, overlay)
        #self.resize(self.img.width(), self.img.height())

    def _draw_rect(self, painter, topleft, bottomright, color):
        painter.setPen(QPen(QBrush(self.config.bb_colors[color]), 5))
        painter.drawRect(QRect(QPoint(*topleft), QPoint(*bottomright)))

    def _new_overlay(self, img):
        # https://www.riverbankcomputing.com/static/Docs/PyQt4/qimage.html
        overlay = QImage(img.width(), img.height(), QImage.Format_ARGB32)
        overlay.fill(QColor(255, 255, 255, 0))
        # https://www.riverbankcomputing.com/static/Docs/PyQt4/qcolor.html
        return overlay

    def _apply_and_show_overlay(self, img, overlay):
        self.img = QImage(img)

        painter = QPainter()
        painter.begin(self.img)
        painter.drawImage(0, 0, overlay)
        painter.end()

        self.img_view.setPixmap(QPixmap.fromImage(self.img))

    def stop_drag(self, x, y):
        self.bounding_boxes.append(
            (self.started_drag, self._apply_bounds(x, y), self.config.active_color))
        self.started_drag = None

    def _apply_bounds(self, x, y):
        # TODO
        return x, y


# https://www.riverbankcomputing.com/static/Docs/PyQt4/qlabel.html
class ImageView(QLabel):
    def __init__(self, *args, **kwargs):
        super(ImageView, self).__init__(*args, **kwargs)
    
    start_drag = pyqtSignal(int, int)
    drag = pyqtSignal(int, int)
    stop_drag = pyqtSignal(int, int)

    def mousePressEvent(self, ev):
        self.start_drag.emit(ev.pos().x(), ev.pos().y())
    
    def mouseMoveEvent(self, ev):
        self.drag.emit(ev.pos().x(), ev.pos().y())

    def mouseReleaseEvent(self, ev):
        self.stop_drag.emit(ev.pos().x(), ev.pos().y())


class AnnotatorConfigurationModel:
    def __init__(self):
        # TODO load configuration from file
        self.n_classes = 2

        # configuration options
        self.bb_colors = [
            QColor(255, 0, 0, 127),
            QColor(0, 255, 0, 127),
            QColor(0, 0, 255, 127),
            # TODO define more colors
        ]

        # configuration
        self.active_color = 0


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec_())