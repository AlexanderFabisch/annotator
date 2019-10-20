import os
import sys
import uuid
from functools import partial
import csv
from PyQt5.QtCore import pyqtSignal, QRect, QPoint, Qt, QObject, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QSpinBox, QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy, QPushButton, QGridLayout, QProgressBar
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QImage, QColor, QBrush, QPen
import numpy as np
import cv2


# TODO
# * shortcuts for video control
# * load and show existing annotations
# * selector for annotation colors
# * display annotation hints, load from config
# * fix annotation overwrite


class MainWindow(QMainWindow):
    def __init__(self, filename, output_path):
        super(MainWindow, self).__init__()
        self.central_widget = CentralWidget(self, filename, output_path)
        self.setCentralWidget(self.central_widget)
        self.resize(1800, 800)  # TODO
        self.show()


class CentralWidget(QWidget):
    def __init__(self, parent, filename, output_path):
        super(CentralWidget, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.annotator_config = AnnotatorConfigurationModel()
        self.annotation = AnnotationModel(filename, output_path, self.annotator_config)

        splitter = QSplitter(Qt.Horizontal)

        self.canvas = ImageCanvas(self, self.annotator_config, self.annotation)
        splitter.addWidget(self.canvas)

        self.tabs = QTabWidget(self)
        self.tabs.setMinimumWidth(450)
        self.config_editor = ConfigurationEditor(self, self.annotator_config)
        self.tabs.addTab(self.config_editor, "Configuration")
        self.ann_editor = AnnotationEditor(self, self.annotation, self.canvas)
        self.tabs.addTab(self.ann_editor, "Annotation")
        self.video_control = VideoControl(self, self.annotation, self.canvas)
        self.tabs.addTab(self.video_control, "Video")
        splitter.addWidget(self.tabs)

        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Maximum)
        policy.setVerticalPolicy(QSizePolicy.Maximum)
        self.canvas.setSizePolicy(policy)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.layout.addWidget(splitter)


class ConfigurationEditor(QWidget):
    def __init__(self, parent, model):
        super(ConfigurationEditor, self).__init__(parent)
        self.model = model
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
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


class AnnotationEditor(QWidget):
    def __init__(self, parent, model, image_view):
        super(AnnotationEditor, self).__init__(parent)
        self.model = model
        self.image_view = image_view

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.selector = QWidget()
        selector_layout = QHBoxLayout()
        self.selector.setLayout(selector_layout)
        self.button_prev = QPushButton("Previous")
        self.button_prev.pressed.connect(self.select_prev)
        selector_layout.addWidget(self.button_prev)
        self.button_next = QPushButton("Next")
        self.button_next.pressed.connect(self.select_next)
        selector_layout.addWidget(self.button_next)
        self.layout.addWidget(self.selector)

        self.button_change_color = QPushButton("Change Color")
        self.button_change_color.pressed.connect(self.change_color)
        self.layout.addWidget(self.button_change_color)

        self.delete = QPushButton("Delete Selection")
        self.delete.pressed.connect(self.delete_selection)
        self.layout.addWidget(self.delete)

        self.save = QPushButton("Save Annotations")
        self.save.pressed.connect(self.save_annotations)
        self.layout.addWidget(self.save)

    def select_prev(self):
        self.model.select_prev()
        self.image_view.update_annotation()

    def select_next(self):
        self.model.select_next()
        self.image_view.update_annotation()

    def change_color(self):
        self.model.change_color_of_selected_annotation()
        self.image_view.update_annotation()

    def delete_selection(self):
        self.model.delete_selection()
        self.image_view.update_annotation()

    def save_annotations(self):
        self.model.save()


class VideoControl(QWidget):
    def __init__(self, parent, annotation, image_view):
        super(VideoControl, self).__init__(parent)
        self.annotation = annotation
        self.image_view = image_view

        self.layout = QGridLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.n_frames_label = QLabel()
        self.n_frames_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.n_frames_label, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(1, self.annotation.n_frames)
        self.layout.addWidget(self.progress_bar, 0, 1)

        self.msecs_per_frame_label = QLabel()
        self.msecs_per_frame_label.setText("%d FPS" % (1.0 / self.annotation.secs_per_frame))
        self.layout.addWidget(self.msecs_per_frame_label, 1, 0)

        self.duration_label = QLabel()
        self.duration_label.setText("%.3f s" % self.annotation.duration())
        self.layout.addWidget(self.duration_label, 1, 1)

        self.button_next_image = QPushButton("Next Frame")
        self.button_next_image.pressed.connect(self.next_image)
        self.layout.addWidget(self.button_next_image, 2, 1)

        # TODO magic numbers
        self.button_skip100 = QPushButton("Skip 100 Frames")
        self.button_skip100.pressed.connect(partial(self.skip, 100))
        self.layout.addWidget(self.button_skip100, 3, 1)

        self.button_skip500 = QPushButton("Skip 500 Frames")
        self.button_skip500.pressed.connect(partial(self.skip, 500))
        self.layout.addWidget(self.button_skip500, 4, 1)

        self.button_skip1800 = QPushButton("Skip 1800 Frames")
        self.button_skip1800.pressed.connect(partial(self.skip, 1800))
        self.layout.addWidget(self.button_skip1800, 5, 1)

        self.button_prev_image = QPushButton("Previous Frames")
        self.button_prev_image.pressed.connect(partial(self.skip, -1))
        self.layout.addWidget(self.button_prev_image, 2, 0)

        self.button_back100 = QPushButton("Go back 100 Frames")
        self.button_back100.pressed.connect(partial(self.skip, -100))
        self.layout.addWidget(self.button_back100, 3, 0)

        self.button_back500 = QPushButton("Go back 500 Frames")
        self.button_back500.pressed.connect(partial(self.skip, -500))
        self.layout.addWidget(self.button_back500, 4, 0)

        self.button_back1800 = QPushButton("Go back 1800 Frames")
        self.button_back1800.pressed.connect(partial(self.skip, -1800))
        self.layout.addWidget(self.button_back1800, 5, 0)

        self.button_play = QPushButton("Play")
        self.layout.addWidget(self.button_play, 6, 0)

        self.button_stop = QPushButton("Stop")
        self.layout.addWidget(self.button_stop, 6, 1)

        self.play_timer = QTimer(self)
        self.button_play.pressed.connect(
            partial(self.play_timer.start, int(self.annotation.secs_per_frame / 1000.0)))
        self.play_timer.timeout.connect(self.next_image)
        self.button_stop.pressed.connect(self.play_timer.stop)

        self.update_info()

    def next_image(self):
        self.annotation.next_image()
        self.image_view.update_image()
        self.update_info()

    def skip(self, frames):
        self.annotation.skip(frames)
        self.image_view.update_image()
        self.update_info()

    def update_info(self):
        self.n_frames_label.setText(
            "%d / %d Frames" % (self.annotation.image_idx + 1,
                                self.annotation.n_frames))
        self.progress_bar.setValue(self.annotation.image_idx + 1)


class ImageCanvas(QWidget):
    def __init__(self, parent, config, annotation):
        super(ImageCanvas, self).__init__(parent)
        self.config = config
        self.annotation = annotation

        self.setWindowTitle("Image annotator")

        self.img_view = ImageView(self)
        self.img_view.start_drag.connect(self.start_drag)
        self.img_view.drag.connect(self.drag)
        self.img_view.stop_drag.connect(self.stop_drag)

        self.update_image()

        # temporary variables
        self.started_drag = None

    def start_drag(self, x, y):
        self.started_drag = self._apply_bounds(x, y)

    def drag(self, x, y):
        overlay = self._new_overlay(self.original_img)
        x, y = self._apply_bounds(x, y)

        # TODO could be done directly, without overlay image
        painter = QPainter()
        painter.begin(overlay)
        i = 0
        for topleft, bottomright, color in self.annotation.bounding_boxes:
            self._draw_rect(
                painter, topleft, bottomright, color,
                selected=self.annotation.selected_annotation == i)
            i += 1
        self._draw_rect(
            painter, self.started_drag, (x, y), self.config.active_color,
            selected=False)
        painter.end()

        self._apply_and_show_overlay(self.original_img, overlay)

    def _draw_rect(self, painter, topleft, bottomright, color, selected=False):
        width = 10 if selected else 5
        painter.setPen(QPen(QBrush(self.config.bb_colors[color]), width))
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
        self.annotation.bounding_boxes.append(
            [self.started_drag, self._apply_bounds(x, y), self.config.active_color])
        self.started_drag = None

    def _apply_bounds(self, x, y):
        return min(max(x, 0), self.img.width()), min(max(y, 0), self.img.height())

    def update_image(self):
        data = self.annotation.image
        self.original_img = QImage(
            data.data, data.shape[1], data.shape[0], 3 * data.shape[1],
            QImage.Format_RGB888).rgbSwapped()

        overlay = self._new_overlay(self.original_img)
        self._apply_and_show_overlay(self.original_img, overlay)

    def update_annotation(self):
        overlay = self._new_overlay(self.original_img)
        painter = QPainter()
        painter.begin(overlay)
        i = 0
        for topleft, bottomright, color in self.annotation.bounding_boxes:
            self._draw_rect(
                painter, topleft, bottomright, color,
                selected=self.annotation.selected_annotation == i)
            i += 1
        painter.end()
        self._apply_and_show_overlay(self.original_img, overlay)


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


class AnnotationModel:  # TODO extract VideoModel?
    def __init__(self, filename, output_path, annotator_config,
                 image_size=(1280, 720)):
        self.filename = filename
        self.output_path = output_path
        self.annotator_config = annotator_config
        self.image_size = image_size
        self.cap = cv2.VideoCapture(self.filename)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.secs_per_frame = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)
        self.image_idx = -1
        self.next_image()

    def duration(self):
        return self.n_frames * self.secs_per_frame

    def next_image(self):
        self.image_idx += 1
        if self.image_idx >= self.n_frames:
            self.image_idx -= 1
            return
        self.reset_annotation()
        self._read_image()

    def skip(self, skip_frames):
        self.reset_annotation()
        self.image_idx += skip_frames
        if self.image_idx >= self.n_frames:
            self.image_idx = self.n_frames - 1
        elif self.image_idx < 0:
            self.image_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.image_idx)
        self._read_image()

    def _read_image(self):
        assert self.cap.isOpened()
        ret, image = self.cap.read()  # TODO ret?
        self.image = cv2.resize(image, self.image_size)

    def reset_annotation(self):
        self.bounding_boxes = []
        self.selected_annotation = None

    def select_prev(self):
        if self.selected_annotation is None:
            if len(self.bounding_boxes) > 0:
                self.selected_annotation = len(self.bounding_boxes) - 1
            else:
                return
        else:
            self.selected_annotation -= 1
            if self.selected_annotation < 0:
                self.selected_annotation = 0
    
    def select_next(self):
        if self.selected_annotation is None:
            if len(self.bounding_boxes) > 0:
                self.selected_annotation = 0
            else:
                return
        else:
            self.selected_annotation += 1
            if self.selected_annotation >= len(self.bounding_boxes):
                self.selected_annotation -= 1
    
    def change_color_of_selected_annotation(self):
        if self.selected_annotation is None:
            return
        self.bounding_boxes[self.selected_annotation][2] = (
            (self.bounding_boxes[self.selected_annotation][2] + 1)
             % self.annotator_config.n_classes)

    def delete_selection(self):
        if self.selected_annotation is None:
            return
        self.bounding_boxes.pop(self.selected_annotation)
        if self.selected_annotation >= len(self.bounding_boxes):
            self.selected_annotation -= 1
        if len(self.bounding_boxes) == 0:
            self.selected_annotation = None

    # TODO load data in memory

    def save(self):  # TODO refactor
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        filename = os.path.join(
            self.output_path,
            "annotated_%s_%08d.jpg" % (uuid.uuid4(), self.image_idx))
        cv2.imwrite(filename, self.image)

        annotations_filename = os.path.join(self.output_path, "annotations.csv")
        if os.path.exists(annotations_filename):
            with open(annotations_filename, "r") as f:
                annotations_reader = csv.reader(f, delimiter=",")
                rows = [r for r in annotations_reader]

            # TODO does not work
            rows_to_delete = []
            for row_idx in range(len(rows)):
                if rows[row_idx][0] == filename and rows[row_idx][1] == self.image_idx:
                    rows_to_delete.append(rows[row_idx])
            for row in rows_to_delete:  # TODO more efficient?
                rows.remove(row)
        else:
            rows = []

        for bb in self.bounding_boxes:
            x_min = min(bb[0][0], bb[1][0])
            x_max = max(bb[0][0], bb[1][0])
            y_min = min(bb[0][1], bb[1][1])
            y_max = max(bb[0][1], bb[1][1])
            rows.append(
                (filename, self.image_idx,
                x_min, y_min, x_max, y_max, bb[2]))

        with open(annotations_filename, "w") as f:
            annotations_writer = csv.writer(f, delimiter=",")
            for row in rows:
                annotations_writer.writerow(row)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # TODO argparser
    win = MainWindow(sys.argv[1], sys.argv[2])
    sys.exit(app.exec_())