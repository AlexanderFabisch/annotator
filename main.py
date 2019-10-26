import os
import sys
import argparse
import hashlib
import yaml
import warnings
from functools import partial
import csv
from PyQt5.QtCore import pyqtSignal, QRect, QPoint, Qt, QObject, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QSpinBox,
    QLabel, QVBoxLayout, QHBoxLayout, QSplitter, QSizePolicy,
    QPushButton, QGridLayout, QProgressBar, QShortcut, QGroupBox)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QImage, QColor, QBrush, QPen,
    QKeySequence, QPalette)
import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self, args):
        super(MainWindow, self).__init__()
        self.central_widget = CentralWidget(self, args)
        self.setCentralWidget(self.central_widget)
        self.resize(1800, 800)  # TODO
        self.show()


class CentralWidget(QWidget):
    def __init__(self, parent, args):
        super(CentralWidget, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.annotator_config = AnnotatorConfigurationModel(args.config)
        self.annotation = AnnotationModel(args.video, args.output, self.annotator_config)

        splitter = QSplitter(Qt.Horizontal)

        self.canvas = ImageCanvas(self, self.annotator_config, self.annotation)
        splitter.addWidget(self.canvas)

        self.sidebar = QWidget(self)
        self.sidebar.setMinimumWidth(450)
        splitter.addWidget(self.sidebar)
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        self.video_control = VideoControl(self, self.annotation, self.canvas)
        self.sidebar_layout.addWidget(self.video_control)
        self.config_editor = ConfigurationEditor(self, self.annotator_config)
        self.sidebar_layout.addWidget(self.config_editor)
        self.ann_editor = AnnotationEditor(self, self.annotation, self.canvas)
        self.sidebar_layout.addWidget(self.ann_editor)

        policy = QSizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Maximum)
        policy.setVerticalPolicy(QSizePolicy.Maximum)
        self.canvas.setSizePolicy(policy)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        self.layout.addWidget(splitter)


class ConfigurationEditor(QGroupBox):
    def __init__(self, parent, model):
        super(ConfigurationEditor, self).__init__("Annotation", parent)
        self.model = model
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.color_display = QLabel(self)
        self.layout.addWidget(self.color_display)
        self.color_image = QImage(420, 20, QImage.Format_ARGB32)

        self.color_selector_description = QLabel(self)
        self.color_selector_description.setText("Select Color (Toggle: Tab)")
        self.layout.addWidget(self.color_selector_description)

        self.color_buttons = []
        self.color_shortcuts = []
        keys = [
            Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5,
            Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9, Qt.Key_0]
        for color_idx in range(self.model.n_classes):
            label = "%s (%s)" % (self.model.classes[color_idx],
                                           QKeySequence(keys[color_idx]).toString())
            button = QPushButton(label)
            button.setStyleSheet("QPushButton { color: white; }")
            palette = button.palette()
            palette.setColor(QPalette.Button, self.model.bb_colors[color_idx])
            button.setAutoFillBackground(True)
            button.setPalette(palette)
            button.update()
            button.pressed.connect(partial(self.change_active_color, color_idx))
            self.layout.addWidget(button)
            self.color_buttons.append(button)

            if color_idx >= len(keys):
                warnings.warn("Too many classes. Cannot assign shortcut to each class.")
                continue
            shortcut = QShortcut(keys[color_idx], self)
            shortcut.activated.connect(partial(self.change_active_color, color_idx))
            self.color_shortcuts.append(shortcut)

        self.shortcut_change_active_color = QShortcut(Qt.Key_Tab, self)
        self.shortcut_change_active_color.activated.connect(self.toggle_active_color)

        self.update_color_info()

    def change_active_color(self, color):
        self.model.active_color = color
        self.update_color_info()

    def toggle_active_color(self):
        self.model.toggle_colors()
        self.update_color_info()

    def update_color_info(self):
        self.color_image.fill(self.model.bb_colors[self.model.active_color])
        self.color_display.setPixmap(QPixmap.fromImage(self.color_image))


class AnnotationEditor(QGroupBox):
    def __init__(self, parent, model, image_view):
        super(AnnotationEditor, self).__init__("Modify Annotations", parent)
        self.model = model
        self.image_view = image_view

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.selector = QWidget()
        selector_layout = QHBoxLayout()
        self.selector.setLayout(selector_layout)
        self.button_prev = QPushButton("Previous (Arrow Up)")
        self.button_prev.pressed.connect(self.select_prev)
        selector_layout.addWidget(self.button_prev)
        self.button_next = QPushButton("Next (Arrow Down)")
        self.button_next.pressed.connect(self.select_next)
        selector_layout.addWidget(self.button_next)
        self.layout.addWidget(self.selector)

        self.button_change_color = QPushButton("Change Color (Enter)")
        self.button_change_color.pressed.connect(self.change_color)
        self.layout.addWidget(self.button_change_color)

        self.delete = QPushButton("Delete Selection (Del)")
        self.delete.pressed.connect(self.delete_selection)
        self.layout.addWidget(self.delete)

        self.save = QPushButton("Save Annotations (Ctrl+s)")
        self.save.pressed.connect(self.save_annotations)
        self.layout.addWidget(self.save)

        self.shortcut_select_prev = QShortcut(Qt.Key_Up, self)
        self.shortcut_select_prev.activated.connect(self.select_prev)

        self.shortcut_select_next = QShortcut(Qt.Key_Down, self)
        self.shortcut_select_next.activated.connect(self.select_next)

        self.shortcut_change_color = QShortcut(Qt.Key_Enter, self)
        self.shortcut_change_color.activated.connect(self.change_color)

        self.shortcut_change_color2 = QShortcut(Qt.Key_Return, self)
        self.shortcut_change_color2.activated.connect(self.change_color)

        self.shortcut_delete = QShortcut(Qt.Key_Delete, self)
        self.shortcut_delete.activated.connect(self.delete_selection)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+s"), self)
        self.shortcut_save.activated.connect(self.save_annotations)

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


class VideoControl(QGroupBox):
    def __init__(self, parent, annotation, image_view):
        super(VideoControl, self).__init__("Video Control", parent)
        self.annotation = annotation
        self.image_view = image_view

        self.layout = QGridLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        self.n_frames_label = QLabel()
        self.n_frames_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.n_frames_label, 0, 0, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(1, self.annotation.video_model.n_frames)
        self.layout.addWidget(self.progress_bar, 1, 0, 1, 2)

        self.msecs_per_frame_label = QLabel()
        self.msecs_per_frame_label.setText(
            "%d FPS" % (1.0 / self.annotation.video_model.secs_per_frame))
        self.layout.addWidget(self.msecs_per_frame_label, 2, 0)

        self.duration_label = QLabel()
        self.duration_label.setText("%.3f s" % self.annotation.video_model.duration())
        self.layout.addWidget(self.duration_label, 2, 1)

        self.button_next_image = QPushButton("Next Frame")
        self.button_next_image.pressed.connect(self.next_image)
        self.layout.addWidget(self.button_next_image, 3, 1)

        # TODO magic numbers
        self.button_skip100 = QPushButton("Skip 100 Frames")
        self.button_skip100.pressed.connect(partial(self.skip, 100))
        self.layout.addWidget(self.button_skip100, 4, 1)

        self.button_skip500 = QPushButton("Skip 500 Frames")
        self.button_skip500.pressed.connect(partial(self.skip, 500))
        self.layout.addWidget(self.button_skip500, 5, 1)

        self.button_skip1800 = QPushButton("Skip 1800 Frames (+)")
        self.button_skip1800.pressed.connect(partial(self.skip, 1800))
        self.layout.addWidget(self.button_skip1800, 6, 1)

        self.button_prev_image = QPushButton("Previous Frames")
        self.button_prev_image.pressed.connect(partial(self.skip, -1))
        self.layout.addWidget(self.button_prev_image, 3, 0)

        self.button_back100 = QPushButton("Go back 100 Frames")
        self.button_back100.pressed.connect(partial(self.skip, -100))
        self.layout.addWidget(self.button_back100, 4, 0)

        self.button_back500 = QPushButton("Go back 500 Frames")
        self.button_back500.pressed.connect(partial(self.skip, -500))
        self.layout.addWidget(self.button_back500, 5, 0)

        self.button_back1800 = QPushButton("Go back 1800 Frames (-)")
        self.button_back1800.pressed.connect(partial(self.skip, -1800))
        self.layout.addWidget(self.button_back1800, 6, 0)

        self.button_play = QPushButton("Play (Space)")
        self.layout.addWidget(self.button_play, 7, 0)

        self.button_stop = QPushButton("Stop (Space)")
        self.layout.addWidget(self.button_stop, 7, 1)

        self.play_timer = QTimer(self)
        self.button_play.pressed.connect(self.play)
        self.play_timer.timeout.connect(self.next_image)
        self.button_stop.pressed.connect(self.stop)

        self.shortcut_play_stop = QShortcut(Qt.Key_Space, self)
        self.shortcut_play_stop.activated.connect(self.toggle_play_stop)

        self.shortcut_skip1800 = QShortcut(Qt.Key_Plus, self)
        self.shortcut_skip1800.activated.connect(partial(self.skip, 1800))

        self.shortcut_back1800 = QShortcut(Qt.Key_Minus, self)
        self.shortcut_back1800.activated.connect(partial(self.skip, -1800))

        self.playing = False

        self.update_info()

    def next_image(self):
        update_required = self.annotation.next_image()
        if not update_required:
            if self.playing:
                self.stop()
            return
        self.image_view.update_image()
        self.image_view.update_annotation()
        self.update_info()

    def skip(self, frames):
        update_required = self.annotation.skip(frames)
        if not update_required:
            return
        self.image_view.update_image()
        self.image_view.update_annotation()
        self.update_info()

    def play(self):
        self.play_timer.start()
        self.playing = True

    def stop(self):
        self.play_timer.stop()
        self.playing = False

    def toggle_play_stop(self):
        if self.playing:
            self.play_timer.stop()
        else:
            self.play_timer.start()
        self.playing = not self.playing

    def update_info(self):
        self.n_frames_label.setText(
            "%d / %d Frames" % (self.annotation.video_model.frame_idx + 1,
                                self.annotation.video_model.n_frames))
        self.progress_bar.setValue(self.annotation.video_model.frame_idx + 1)


class ImageCanvas(QGroupBox):
    def __init__(self, parent, config, annotation):
        super(ImageCanvas, self).__init__("Video", parent)
        self.config = config
        self.annotation = annotation

        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.setLayout(self.layout)

        self.setWindowTitle("Image annotator")

        self.img_view = ImageView()
        self.img_view.start_drag.connect(self.start_drag)
        self.img_view.drag.connect(self.drag)
        self.img_view.stop_drag.connect(self.stop_drag)
        self.layout.addWidget(self.img_view)

        # temporary variables
        self.started_drag = None
        self.overlay = QImage(
            self.config.image_size[0], self.config.image_size[1],
            QImage.Format_ARGB32)

        self.update_image()
        self.update_annotation()

    def start_drag(self, x, y):
        self.started_drag = self._apply_bounds(x, y)

    def drag(self, x, y):
        self._reset_overlay()
        x, y = self._apply_bounds(x, y)

        painter = QPainter()
        painter.begin(self.overlay)
        self._paint_bbs(painter)
        self._draw_rect(
            painter, self.started_drag, (x, y), self.config.active_color,
            selected=False)
        painter.end()

        self._apply_and_show_overlay()

    def stop_drag(self, x, y):
        self.annotation.bounding_boxes.append(
            [self.started_drag, self._apply_bounds(x, y), self.config.active_color])
        self.started_drag = None

    def _apply_bounds(self, x, y):
        return min(max(x, 0), self.img.width()), min(max(y, 0), self.img.height())

    def update_image(self):
        data = self.annotation.video_model.image
        # TODO can we reuse the image from previous frames?
        self.original_img = QImage(
            data.data, data.shape[1], data.shape[0], 3 * data.shape[1],
            QImage.Format_RGB888).rgbSwapped()

    def update_annotation(self):
        self._reset_overlay()
        painter = QPainter()
        painter.begin(self.overlay)
        self._paint_bbs(painter)
        painter.end()
        self._apply_and_show_overlay()

    def _paint_bbs(self, painter):
        i = 0
        for topleft, bottomright, color in self.annotation.bounding_boxes:
            self._draw_rect(
                painter, topleft, bottomright, color,
                selected=self.annotation.selected_annotation == i)
            i += 1

    def _draw_rect(self, painter, topleft, bottomright, color, selected=False):
        width = 10 if selected else 5
        painter.setPen(QPen(QBrush(self.config.bb_colors[color]), width))
        painter.drawRect(QRect(QPoint(*topleft), QPoint(*bottomright)))

    def _reset_overlay(self):
        self.overlay.fill(QColor(255, 255, 255, 0))

    def _apply_and_show_overlay(self):
        self.img = self.original_img.copy()

        painter = QPainter()
        painter.begin(self.img)
        painter.drawImage(0, 0, self.overlay)
        painter.end()

        self.img_view.setPixmap(QPixmap.fromImage(self.img))


class ImageView(QLabel):
    def __init__(self):
        super(ImageView, self).__init__()
    
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
    def __init__(self, configfile):
        config = {"classes": ["Class 1", "Class 2"],
                  "resolution": (1280, 720)}
        if configfile is not None:
            with open(configfile, "r") as f:
                config = yaml.safe_load(f)
            if "classes" not in config:
                raise Exception("Could not find class names")

        # TODO validate values
        self.n_classes = len(config["classes"])
        self.classes = config["classes"]
        self.image_size = tuple(config["resolution"])

        self.bb_colors = [
            QColor(30, 45, 69),
            QColor(87, 52, 32),
            QColor(33, 66, 41),
            QColor(77, 31, 32),
            QColor(51, 45, 70),
            QColor(58, 47, 38),
            QColor(85, 55, 76),
            QColor(55, 55, 55),
            QColor(80, 73, 45),
            QColor(39, 71, 80),
            #QColor(255, 0, 0, 127),
            #QColor(0, 255, 0, 127),
            #QColor(0, 0, 255, 127),
            # TODO define more colors
        ]

        self.active_color = 0

    def toggle_colors(self):
        self.active_color = (self.active_color + 1) % self.n_classes


class AnnotationModel:
    def __init__(self, filename, output_path, annotator_config):
        self.video_model = VideoModel(filename, annotator_config.image_size)
        self.output_path = output_path
        self.annotator_config = annotator_config
        self.image_idx = -1
        self.image_filename = None
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.annotations_filename = os.path.join(
            self.output_path, "annotations.csv")
        self.load()
        self.bounding_boxes = []
        self.next_image()

    def next_image(self):
        self.reset_annotation()
        last_frame_idx = self.image_idx
        self.image_idx = self.video_model.next_frame()
        self._update_image_filename()
        self._load_annotation_of_current_image()
        return last_frame_idx != self.image_idx

    def skip(self, skip_frames):
        self.reset_annotation()
        last_frame_idx = self.image_idx
        self.image_idx = self.video_model.jump(skip_frames)
        self._update_image_filename()
        self._load_annotation_of_current_image()
        return last_frame_idx != self.image_idx

    def reset_annotation(self):
        self._save_annotations_as_rows()
        self.bounding_boxes = []
        self.selected_annotation = None

    def _save_annotations_as_rows(self):
        self._remove_entries_from_this_image()
        for bb in self.bounding_boxes:
            x_min = min(bb[0][0], bb[1][0])
            x_max = max(bb[0][0], bb[1][0])
            y_min = min(bb[0][1], bb[1][1])
            y_max = max(bb[0][1], bb[1][1])
            row = (self.image_filename, self.image_idx,
                   x_min, y_min, x_max, y_max, bb[2])
            self.rows.append(row)

        if (len(self.bounding_boxes) > 0 and
                self.image_filename is not None and
                not os.path.exists(self.image_filename)):
            self.video_model.buffer_frame(self.image_filename)

    def _update_image_filename(self):
        m = hashlib.md5()
        m.update(self.video_model.filename.encode())
        identifier = m.hexdigest()
        self.image_filename = os.path.join(
            self.output_path,
            "annotated_%s_%08d.jpg" % (identifier, self.image_idx))

    def _load_annotation_of_current_image(self):
        for row in self.rows:
            if row[0] == self.image_filename and row[1] == self.image_idx:
                x_min, y_min, x_max, y_max, color = row[2:]
                self.bounding_boxes.append([[x_min, y_min], [x_max, y_max], color])

    def select_prev(self):
        if self.selected_annotation is None:
            if len(self.bounding_boxes) > 0:
                self.selected_annotation = len(self.bounding_boxes) - 1
            else:
                return
        else:
            self.selected_annotation -= 1
            if self.selected_annotation < 0:
                self.selected_annotation = len(self.bounding_boxes) - 1
    
    def select_next(self):
        if self.selected_annotation is None:
            if len(self.bounding_boxes) > 0:
                self.selected_annotation = 0
            else:
                return
        else:
            self.selected_annotation += 1
            if self.selected_annotation >= len(self.bounding_boxes):
                self.selected_annotation = 0
    
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

    def load(self):
        if os.path.exists(self.annotations_filename):
            with open(self.annotations_filename, "r") as f:
                annotations_reader = csv.reader(f, delimiter=",")
                self.rows = [self._convert_row(r) for r in annotations_reader]
        else:
            self.rows = []

    def _convert_row(self, row):
        return [row[0]] + list(map(int, map(float, row[1:])))

    def _remove_entries_from_this_image(self):
        rows_to_delete = []
        for row_idx in range(len(self.rows)):
            if self.rows[row_idx][0] == self.image_filename and self.rows[row_idx][1] == self.image_idx:
                rows_to_delete.append(self.rows[row_idx])
        for row in rows_to_delete:  # TODO more efficient?
            self.rows.remove(row)

    def save(self):
        if len(self.bounding_boxes) > 0 and not os.path.exists(self.image_filename):
            self.video_model.buffer_frame(self.image_filename)
        self._save_annotations_as_rows()

        self.video_model.write_buffer()

        with open(self.annotations_filename, "w") as f:
            annotations_writer = csv.writer(f, delimiter=",")
            for row in self.rows:
                annotations_writer.writerow(row)


class VideoModel:
    def __init__(self, filename, image_size):
        self.filename = filename
        self.image_size = image_size
        self.cap = cv2.VideoCapture(self.filename)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.secs_per_frame = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_idx = -1
        self.frame_buffer = {}
        # TODO VideoCapture.release() on shutdown

    def duration(self):
        return self.n_frames * self.secs_per_frame

    def next_frame(self):
        self.frame_idx += 1
        success = self._read_image()
        if not success:
            self.frame_idx -= 1
        return self.frame_idx

    def jump(self, skip_frames):
        last_frame_idx = self.frame_idx
        self.frame_idx += skip_frames
        if self.frame_idx >= self.n_frames:
            self.frame_idx = self.n_frames - 1
        elif self.frame_idx < 0:
            self.frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)

        success = self._read_image()
        if not success:
            self.frame_idx = last_frame_idx
        return self.frame_idx

    def _read_image(self):
        assert self.cap.isOpened()
        success, image = self.cap.read()
        if success:
            self.image = cv2.resize(image, self.image_size)
        return success

    def buffer_frame(self, filename):
        self.frame_buffer[self.frame_idx] = (filename, self.image)

    def write_buffer(self):
        for filename, image in self.frame_buffer.values():
            cv2.imwrite(filename, image)
        self.frame_buffer = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Annotator")
    parser.add_argument("video", help="Location of the video file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument(
        "--config", nargs="?", default=None,
        help="Configuration file for annotator")
    return parser.parse_args()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    args = parse_args()
    win = MainWindow(args)
    sys.exit(app.exec_())