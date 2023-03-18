# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setting.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_setting_Window(object):
    def setupUi(self, setting_Window):
        setting_Window.setObjectName("setting_Window")
        setting_Window.resize(397, 473)
        self.centralwidget = QtWidgets.QWidget(setting_Window)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_setting_reset = QtWidgets.QPushButton(self.centralwidget)
        self.btn_setting_reset.setGeometry(QtCore.QRect(40, 330, 331, 24))
        self.btn_setting_reset.setObjectName("btn_setting_reset")
        self.label_setting_output_path = QtWidgets.QLabel(self.centralwidget)
        self.label_setting_output_path.setGeometry(QtCore.QRect(40, 100, 331, 31))
        self.label_setting_output_path.setObjectName("label_setting_output_path")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(40, 20, 321, 36))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.checkBox_save_vid = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(12)
        self.checkBox_save_vid.setFont(font)
        self.checkBox_save_vid.setObjectName("checkBox_save_vid")
        self.horizontalLayout_3.addWidget(self.checkBox_save_vid)
        self.checkBox_save_txt = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.checkBox_save_txt.setFont(font)
        self.checkBox_save_txt.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.checkBox_save_txt.setAutoFillBackground(False)
        self.checkBox_save_txt.setObjectName("checkBox_save_txt")
        self.horizontalLayout_3.addWidget(self.checkBox_save_txt)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 140, 321, 32))
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 210, 321, 32))
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(16)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.SpinBox_iou_setting = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.SpinBox_iou_setting.setGeometry(QtCore.QRect(40, 180, 58, 22))
        self.SpinBox_iou_setting.setObjectName("SpinBox_iou_setting")
        self.slider_iou_setting = QtWidgets.QSlider(self.centralwidget)
        self.slider_iou_setting.setGeometry(QtCore.QRect(100, 180, 260, 22))
        self.slider_iou_setting.setMaximum(100)
        self.slider_iou_setting.setOrientation(QtCore.Qt.Horizontal)
        self.slider_iou_setting.setObjectName("slider_iou_setting")
        self.SpinBox_conf_setting = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.SpinBox_conf_setting.setGeometry(QtCore.QRect(40, 250, 58, 22))
        self.SpinBox_conf_setting.setObjectName("SpinBox_conf_setting")
        self.slider_conf_setting = QtWidgets.QSlider(self.centralwidget)
        self.slider_conf_setting.setGeometry(QtCore.QRect(100, 250, 260, 22))
        self.slider_conf_setting.setMaximum(100)
        self.slider_conf_setting.setOrientation(QtCore.Qt.Horizontal)
        self.slider_conf_setting.setObjectName("slider_conf_setting")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 280, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3.setIndent(0)
        self.label_3.setObjectName("label_3")
        self.setting_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.setting_comboBox.setGeometry(QtCore.QRect(190, 290, 171, 22))
        self.setting_comboBox.setObjectName("setting_comboBox")
        self.btn_setting_submit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_setting_submit.setEnabled(True)
        self.btn_setting_submit.setGeometry(QtCore.QRect(40, 400, 162, 31))
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        self.btn_setting_submit.setFont(font)
        self.btn_setting_submit.setObjectName("btn_setting_submit")
        self.btn_setting_cancle = QtWidgets.QPushButton(self.centralwidget)
        self.btn_setting_cancle.setGeometry(QtCore.QRect(210, 400, 161, 31))
        self.btn_setting_cancle.setObjectName("btn_setting_cancle")
        self.btn_setting_folder = QtWidgets.QPushButton(self.centralwidget)
        self.btn_setting_folder.setGeometry(QtCore.QRect(40, 60, 321, 24))
        self.btn_setting_folder.setObjectName("btn_setting_folder")
        setting_Window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(setting_Window)
        self.statusbar.setObjectName("statusbar")
        setting_Window.setStatusBar(self.statusbar)

        self.retranslateUi(setting_Window)
        QtCore.QMetaObject.connectSlotsByName(setting_Window)

    def retranslateUi(self, setting_Window):
        _translate = QtCore.QCoreApplication.translate
        setting_Window.setWindowTitle(_translate("setting_Window", "Setting"))
        self.btn_setting_reset.setText(_translate("setting_Window", "reset default"))
        self.label_setting_output_path.setText(_translate("setting_Window", "TextLabel"))
        self.checkBox_save_vid.setText(_translate("setting_Window", "Save video"))
        self.checkBox_save_txt.setText(_translate("setting_Window", "Save txt"))
        self.label_4.setText(_translate("setting_Window", "IOU threshold for NMS"))
        self.label_5.setText(_translate("setting_Window", "Object confidence threshold"))
        self.label_3.setText(_translate("setting_Window", "Yolo model"))
        self.btn_setting_submit.setText(_translate("setting_Window", "submit"))
        self.btn_setting_cancle.setText(_translate("setting_Window", "cancle"))
        self.btn_setting_folder.setText(_translate("setting_Window", "Folder Selection"))
