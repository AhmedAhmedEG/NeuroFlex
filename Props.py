import math
import time

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QMainWindow, QProgressBar, \
    QApplication
from PySide6.QtCore import Signal, Qt, QSize, QPoint, QPropertyAnimation, QEasingCurve, QObject, QEvent, \
    QParallelAnimationGroup, QRect

from PySide6.QtGui import QFont, QIcon, QPixmap, QGuiApplication, QPainter, QPen, QPalette, QColor
from PySide6 import QtGui
import dataclasses
import numpy as np
import cv2

RIGHT_ARM = [12, 14, 16]
LEFT_ARM = [11, 13, 15]
LEFT_LEG = [23, 25, 27]
RIGHT_LEG = [24, 26, 28]

THUMB = [0, 1, 2, 3, 4]
INDEX = [0, 5, 6, 7, 8]
MIDDLE = [0, 9, 10, 11, 12]
RING = [0, 13, 14, 15, 16]
PINKY = [0, 17, 18, 19, 20]


class ModernTitleBar(QFrame):
    closed = Signal()

    def __init__(self, title='', icon=None, movable=True, closable=True, maximizable=False, minimizable=True):
        super().__init__()
        self.setObjectName('ModernTitleBar')
        self.setFixedHeight(30)

        self.movable = movable

        self.minimize_size = None
        self.window_offset = None
        self.font_size = 11

        # Structure
        self.body = QHBoxLayout()
        self.body.setSpacing(5)
        self.body.setContentsMargins(6, 2, 4, 0)

        # Components
        self.icon = QLabel()
        if icon is not None:
            self.set_icon(icon)

        else:
            self.icon.hide()

        f = QFont()
        f.setBold(True)
        f.setFamily('calibri')
        f.setPointSize(self.font_size)

        self.title = QLabel(title)
        self.title.setFont(f)

        self.minimize_btn = QPushButton('–')
        self.minimize_btn.setFixedWidth(20)
        self.minimize_btn.setFlat(True)
        self.minimize_btn.setFont(f)

        self.maximize_btn = QPushButton('❒')
        self.maximize_btn.setFixedWidth(20)
        self.maximize_btn.setFlat(True)
        self.maximize_btn.setFont(f)

        self.exit_btn = QPushButton('X')
        self.exit_btn.setFixedWidth(20)
        self.exit_btn.setFlat(True)
        self.exit_btn.setFont(f)

        # Assembly
        self.body.addWidget(self.icon, alignment=Qt.AlignLeft)
        self.body.addWidget(self.title, alignment=Qt.AlignLeft)

        self.body.addStretch(1)

        if minimizable:
            self.body.addWidget(self.minimize_btn, alignment=Qt.AlignRight | Qt.AlignmentFlag.AlignTop)

        if maximizable:
            self.body.addWidget(self.maximize_btn, alignment=Qt.AlignRight)

        if closable:
            self.body.addWidget(self.exit_btn, alignment=Qt.AlignRight)

        # self.body.addWidget(HSeparator(height=2), 1, 0, 1, 4)
        self.setLayout(self.body)

        # Functionality
        self.minimize_btn.clicked.connect(self.minimize)
        self.maximize_btn.clicked.connect(self.maximize)
        self.exit_btn.clicked.connect(self.exit)

    def set_title(self, title: str) -> None:
        self.title.setText(title)
        self.parent().setWindowTitle(title)

    def set_icon(self, icon: str):
        self.parent().setWindowIcon(QIcon(icon))

        self.icon.setPixmap(QPixmap(icon).scaled(15, 15, mode=Qt.SmoothTransformation))
        self.icon.resize(QSize(15, 15))
        self.icon.show()

    def minimize(self):
        self.parent().showMinimized()

    def maximize(self):

        if self.parent().size() == QGuiApplication.screens()[0].size():
            self.parent().resize(self.minimize_size)
            self.parent().move(
                QGuiApplication.screens()[0].geometry().center() - self.parent().frameGeometry().center())

        else:
            self.minimize_size = self.parent().size()

            self.parent().resize(QGuiApplication.screens()[0].size())
            self.parent().move(QPoint(0, 0))

    def exit(self):
        self.closed.emit()
        self.parent().hide()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(e)

        if not self.movable:
            return

        self.window_offset = e.position()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(e)

        if not self.movable:
            return

        pos = QPoint(e.globalPosition().x() - self.window_offset.x(), e.globalPosition().y() - self.window_offset.y())

        self.pos_anim = QPropertyAnimation(self.parent(), b'pos')
        self.pos_anim.setEndValue(pos)
        self.pos_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
        self.pos_anim.setDuration(400)

        self.pos_anim.start()


class ModernMainWindow(QFrame):
    animation_starting = Signal()
    animation_finished = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        self.anim_duration = 400
        self.stretch_direction = None
        self.cursor_changed = False
        self.anchor = None

        # Structure
        self.detector_body = QVBoxLayout()
        self.detector_body.setSpacing(0)
        self.detector_body.setContentsMargins(0, 0, 0, 0)

        # Components
        self.window_frame = ModernTitleBar(title='')
        self.main_window = QMainWindow()

        # Assembly
        self.detector_body.addWidget(self.window_frame)
        self.detector_body.addWidget(self.main_window, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(self.detector_body)

    def eventFilter(self, watched: QObject, e: QEvent) -> bool:

        if isinstance(watched, QProgressBar):
            return False

        if e.type() == QEvent.Type.MouseMove:
            self.check_corners(self.mapFromGlobal(e.globalPosition()))

            if self.anchor:
                self.mouseMoveEvent(e)
                return True

            else:
                return False

        if e.type() == QEvent.Type.MouseButtonPress and self.stretch_direction:
            self.mousePressEvent(e)
            return False

        if e.type() == QEvent.Type.MouseButtonRelease and self.stretch_direction:
            self.mouseReleaseEvent(e)
            return False

        return False

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(e)

        if self.stretch_direction:
            self.anchor = e.globalPosition()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(e)

        if e.button() != Qt.MouseButtons.LeftButton:
            return

        self.anchor = None
        self.stretch_direction = None
        self.animation_finished.emit()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(e)
        self.check_corners(e.position())

        if self.anchor:
            self.animation_starting.emit()

            if self.stretch_direction == 'r':
                delta = e.globalPosition().x() - (self.width() + self.pos().x())

                self.size_anim = QPropertyAnimation(self, b"size")
                self.size_anim.setEndValue(QSize(self.width() + delta, self.height()))
                self.size_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                self.size_anim.setDuration(self.anim_duration)
                self.size_anim.start()

            elif self.stretch_direction == 'l':
                delta = self.pos().x() - e.globalPosition().x()

                if self.width() + delta < self.minimumWidth():
                    return

                self.animation = QParallelAnimationGroup()

                pos_anim = QPropertyAnimation(self, b'pos')
                pos_anim.setEndValue(QPoint(self.pos().x() - delta, self.pos().y()))
                pos_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                pos_anim.setDuration(self.anim_duration)

                size_anim = QPropertyAnimation(self, b"size")
                size_anim.setEndValue(QSize(self.width() + delta, self.height()))
                size_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                size_anim.setDuration(self.anim_duration)

                self.animation.addAnimation(pos_anim)
                self.animation.addAnimation(size_anim)

                self.animation.start()

            elif self.stretch_direction == 'u':
                delta = self.pos().y() - e.globalPosition().y()

                if self.height() + delta < self.minimumHeight():
                    return

                self.animation = QParallelAnimationGroup()

                pos_anim = QPropertyAnimation(self, b'pos')
                pos_anim.setEndValue(QPoint(self.pos().x(), self.pos().y() - delta))
                pos_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                pos_anim.setDuration(self.anim_duration)

                size_anim = QPropertyAnimation(self, b"size")
                size_anim.setEndValue(QSize(self.width(), self.height() + delta))
                size_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                size_anim.setDuration(self.anim_duration)

                self.animation.addAnimation(pos_anim)
                self.animation.addAnimation(size_anim)

                self.animation.start()

            elif self.stretch_direction == 'd':
                delta = e.globalPosition().y() - (self.height() + self.pos().y())

                self.size_anim = QPropertyAnimation(self, b"size")
                self.size_anim.setEndValue(QSize(self.width(), self.height() + delta))
                self.size_anim.setEasingCurve(QEasingCurve.Type.OutExpo)
                self.size_anim.setDuration(self.anim_duration)
                self.size_anim.start()

    def check_corners(self, pos: QPoint):
        cx = self.width() / 2
        cy = self.height() / 2

        vertical_direction = 'u' if pos.y() < cy else 'd'
        horizontal_direction = 'l' if pos.x() < cx else 'r'

        horizontal_distance = self.width() - pos.x() if horizontal_direction == 'r' else pos.x()
        vertical_distance = self.height() - pos.y() if vertical_direction == 'd' else pos.y()

        if abs(min(horizontal_distance, vertical_distance)) <= 4:

            if horizontal_distance < vertical_distance:

                if not self.cursor_changed:
                    QApplication.setOverrideCursor(Qt.CursorShape.SizeHorCursor)
                    self.cursor_changed = True

                self.stretch_direction = horizontal_direction

            else:

                if not self.cursor_changed:
                    QApplication.setOverrideCursor(Qt.CursorShape.SizeVerCursor)
                    self.cursor_changed = True

                self.main_window.setFocus()
                self.stretch_direction = vertical_direction

        else:

            if not self.anchor:

                if self.cursor_changed:
                    QApplication.restoreOverrideCursor()
                    self.cursor_changed = False

                self.stretch_direction = None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        painter.setPen(QPen(self.palette().window().color(), 4))
        painter.setBrush(self.palette().window())

        drawing_rect = QRect(2, self.window_frame.height() - 2, self.width() - 4, self.height() - self.window_frame.height())
        painter.drawRoundedRect(drawing_rect, 4, 4)


@dataclasses.dataclass
class TherapyLevel:
    left_arm: bool = False
    left_hand: bool = False
    right_arm: bool = False
    right_hand: bool = False

    pose_angles: dict = dataclasses.field(default_factory=dict)
    left_hand_angles: dict = dataclasses.field(default_factory=dict)
    left_palm_orientation: dict = dataclasses.field(default_factory=dict)
    right_hand_angles: dict = dataclasses.field(default_factory=dict)
    right_palm_orientation: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def flatten(*args):
        angles = []

        for arg in args:

            for v in arg.values():

                if type(v) in (list, tuple):
                    angles.extend(v)

                else:
                    angles.append(v)

        return angles


class ScoreEvaluator:

    def __init__(self, threshold):
        self.scores = []
        self.threshold = threshold

    def evaluate(self, score):

        if len(self.scores) > 5:
            self.scores.pop(0)

        self.scores.append(score)
        return np.average(self.scores) >= self.threshold


class ScoreTrigger:

    def __init__(self, target, deadline):
        self.deadline = deadline
        self.target = target
        self.previous_time = time.time()

    def check(self, value):

        if value != self.target:
            self.previous_time = time.time()

        if time.time() - self.previous_time >= self.deadline:
            self.previous_time = time.time()
            return True

        else:
            return False


def get_palette():
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor('#cbcbcb'))
    palette.setColor(QPalette.WindowText, QColor('#000000'))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#808080'))

    palette.setColor(QPalette.Button, QColor('#cfcfcf'))  # Changes the background color of QPushButton, QTabBar and QTabWidget.
    palette.setColor(QPalette.ButtonText, QColor('#000000'))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#808080'))

    palette.setColor(QPalette.Base, QColor('#dddddd'))  # Changes background color of text input widgets.

    palette.setColor(QPalette.Highlight, QColor('#2a82da'))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor('#808080'))

    palette.setColor(QPalette.HighlightedText, QColor('#dddddd'))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor('#808080'))

    palette.setColor(QPalette.Text, QColor('#000000'))  # Changes GroupBox, LineEdit and QDocument text color. Also Changes Checkbox check mark color.
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor('#808080'))

    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)

    return palette


def resize_frame(frame, width=None, height=None):

    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]
    aspect_ratio = w / h

    if width is not None:
        return cv2.resize(frame, (width, int(width / aspect_ratio)))

    else:
        return cv2.resize(frame, (int(height * aspect_ratio), height))


def draw_pose_landmarks(frame, points, color=(255, 0, 0)):
    draw_line(frame, [points[l[0]] for l in [LEFT_ARM, RIGHT_ARM, RIGHT_LEG, LEFT_LEG, LEFT_ARM]], color)
    draw_line(frame, [points[i] for i in RIGHT_LEG], color)
    draw_line(frame, [points[i] for i in LEFT_LEG], color)

    draw_points(frame, [points[i] for i in [LEFT_ARM[0], RIGHT_ARM[0], *RIGHT_LEG, *LEFT_LEG]], 3, (255, 255, 255), color)


def draw_arm_landmarks(frame, points, color=(255, 0, 0), right_arm=True, left_arm=True):

    if right_arm:
        draw_line(frame, [points[i] for i in RIGHT_ARM], color)
        draw_points(frame, [points[i] for i in RIGHT_ARM[1:]], 3, (255, 255, 255), color)

    if left_arm:
        draw_line(frame, [points[i] for i in LEFT_ARM], color)
        draw_points(frame, [points[i] for i in LEFT_ARM[1:]], 3, (255, 255, 255), color)


def draw_hand_landmarks(frame, points, color=(255, 0, 0)):
    draw_line(frame, [points[i] for i in THUMB], color)
    draw_line(frame, [points[i] for i in INDEX], color)
    draw_line(frame, [points[i] for i in MIDDLE], color)
    draw_line(frame, [points[i] for i in RING], color)
    draw_line(frame, [points[i] for i in PINKY], color)

    draw_points(frame, [points[i] for i in (THUMB + INDEX + MIDDLE + RING + PINKY)], 3, (255, 255, 255), color)


def draw_line(img, points, color, thickness=2):

    for p1, p2 in zip(points[:-1], points[1:]):

        if None not in [p1, p2]:
            cv2.line(img, p1, p2, color, thickness)


def draw_points(img, points, radius, outer_color, inner_color, thickness=2):

    for p in points:

        if p is not None:
            cv2.circle(img, p, radius, outer_color, thickness)
            cv2.circle(img, p, radius - 1, inner_color, thickness)


def calc_points(landmarks, w, h, visibility=False, use_z=False):
    points = {}

    for i, landmark in enumerate(landmarks):

        if visibility and landmark.visibility < 0.3:
            points[i] = None
            continue

        x, y = min(math.floor(landmark.x * w), w - 1), min(math.floor(landmark.y * h), h - 1)

        if use_z:
            z = min(math.floor(landmark.z * w), w - 1)
            points[i] = (x, y, z)

        else:
            points[i] = (x, y)

    return points


def calc_pose_angles(pose_points):

    if not pose_points:
        return {}

    right_arm_points = [pose_points[i] for i in RIGHT_ARM]
    left_arm_points = [pose_points[i] for i in LEFT_ARM]
    right_leg_points = [pose_points[i] for i in RIGHT_LEG]
    left_leg_points = [pose_points[i] for i in LEFT_LEG]

    return {'Right Shoulder': calc_angles([right_leg_points[0], *right_arm_points[:2]])[0],
            'Right Elbow': calc_angles(right_arm_points)[0],
            'Left Shoulder': calc_angles([left_leg_points[0], *left_arm_points[:2]])[0],
            'Left Elbow': calc_angles(left_arm_points)[0]}


def calc_hand_angles(hand_points):

    if not hand_points:
        return {}

    thumb_points = [hand_points[i] for i in THUMB]
    index_points = [hand_points[i] for i in INDEX]
    middle_points = [hand_points[i] for i in MIDDLE]
    ring_points = [hand_points[i] for i in RING]
    little_points = [hand_points[i] for i in PINKY]

    return {'Thumb': calc_angles(thumb_points),
            'Index': calc_angles(index_points),
            'Middle': calc_angles(middle_points),
            'Ring': calc_angles(ring_points),
            'Little': calc_angles(little_points),
            'ThumbIndex': calc_angle(thumb_points[-2:], index_points[-2:]),
            'IndexMiddle': calc_angle(index_points[-2:], middle_points[-2:]),
            'MiddleRing': calc_angle(middle_points[-2:], ring_points[-2:]),
            'RingLittle': calc_angle(ring_points[-2:], little_points[-2:])}


def calc_palm_orientation(hand_3d_points):

    if not hand_3d_points:
        return ()

    return {'Orientation': calc_orientation(hand_3d_points[MIDDLE[0]], hand_3d_points[MIDDLE[1]]) +
                           calc_orientation(hand_3d_points[INDEX[1]], hand_3d_points[PINKY[1]])}


def calc_pose_orientation(pose_3d_points):

    if not pose_3d_points:
        return {}

    right_arm_points = [pose_3d_points[i] for i in RIGHT_ARM]
    left_arm_points = [pose_3d_points[i] for i in LEFT_ARM]

    return {'Right Orientation': calc_orientation(right_arm_points[0], right_arm_points[1]) +
                                 calc_orientation(right_arm_points[1], right_arm_points[2]),

            'Left Orientation': calc_orientation(left_arm_points[0], left_arm_points[1]) +
                                calc_orientation(left_arm_points[1], left_arm_points[2])}


def calc_angle(a_points, b_points):

    if None not in [*a_points, *b_points]:
        a, b = np.subtract(*reversed(a_points)), np.subtract(*reversed(b_points))
        angle_radians = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        angle_degrees = ((angle_radians * 180) / np.pi)

        # cross_product = np.cross(a, b)
        # if cross_product > 0:
        #     angle_degrees = angle_degrees + 180 if cross_product < 0 else angle_degrees
        # else:
        #     angle_degrees = angle_degrees + 180 if cross_product > 0 else angle_degrees

        if np.isnan(angle_degrees):
            angle_degrees = 0

        return int(angle_degrees)


def calc_angles(points):
    angles = []

    for p1, p2, p3 in zip(points[:-2], points[1:-1], points[2:]):

        if None in [p1, p2, p3]:
            angles.append(None)

        else:
            a, b = np.subtract(p2, p1), np.subtract(p3, p2)
            angle_radians = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            angle_degrees = 180 - ((angle_radians * 180) / np.pi)

            # cross_product = np.cross(a, b)
            # if cross_product > 0:
            #     angle_degrees = 180 - angle_degrees + 180 if cross_product < 0 else angle_degrees
            # else:
            #     angle_degrees = 180 - angle_degrees + 180 if cross_product > 0 else angle_degrees

            if not np.isfinite(angle_degrees):
                angle_degrees = 0

            angles.append(int(angle_degrees))

    return angles


def calc_orientation(p1, p2):

    if None in [p1, p2]:
        return ()

    x, y, z = np.subtract(p2, p1)

    roll = math.atan2(y, z)
    pitch = math.atan2(-x, math.sqrt(y ** 2 + z ** 2))
    yaw = math.atan2(y, x)

    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return int(roll), int(pitch), int(yaw)

