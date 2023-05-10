# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1275, 720)
        font = QtGui.QFont()
        font.setFamily("Prompt Bold")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 230, 241, 381))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.listWidget = QtWidgets.QListWidget(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Poppins Medium")
        font.setPointSize(8)
        self.listWidget.setFont(font)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(290, 70, 961, 591))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.displayImage = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.displayImage.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.displayImage.setText("")
        self.displayImage.setAlignment(QtCore.Qt.AlignCenter)
        self.displayImage.setObjectName("displayImage")
        self.verticalLayout_4.addWidget(self.displayImage)
        self.btn_Start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Start.setGeometry(QtCore.QRect(30, 620, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins SemiBold")
        font.setPointSize(18)
        self.btn_Start.setFont(font)
        self.btn_Start.setObjectName("btn_Start")
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(290, 20, 951, 41))
        font = QtGui.QFont()
        font.setFamily("Prompt SemiBold")
        font.setPointSize(25)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 150, 241, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btn_submit = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_submit.setObjectName("btn_submit")
        self.horizontalLayout_3.addWidget(self.btn_submit)
        self.btn_cancle = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_cancle.setObjectName("btn_cancle")
        self.horizontalLayout_3.addWidget(self.btn_cancle)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(30, 670, 1221, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(30, 620, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins SemiBold")
        font.setPointSize(18)
        self.btn_stop.setFont(font)
        self.btn_stop.setObjectName("btn_stop")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(30, 110, 241, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_creat_area = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_creat_area.setFont(font)
        self.btn_creat_area.setObjectName("btn_creat_area")
        self.horizontalLayout.addWidget(self.btn_creat_area)
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(30, 70, 241, 41))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.btn_open = QtWidgets.QPushButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_open.setFont(font)
        self.btn_open.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("pyqt_icon/folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_open.setIcon(icon)
        self.btn_open.setIconSize(QtCore.QSize(30, 30))
        self.btn_open.setObjectName("btn_open")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btn_open)
        self.label_fielname = QtWidgets.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Poppins Medium")
        font.setPointSize(13)
        self.label_fielname.setFont(font)
        self.label_fielname.setObjectName("label_fielname")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_fielname)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(30, 190, 241, 41))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.btn_resetArea = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.btn_resetArea.setObjectName("btn_resetArea")
        self.horizontalLayout_5.addWidget(self.btn_resetArea)
        self.btn_undo = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.btn_undo.setObjectName("btn_undo")
        self.horizontalLayout_5.addWidget(self.btn_undo)
        self.btn_setting = QtWidgets.QPushButton(self.centralwidget)
        self.btn_setting.setGeometry(QtCore.QRect(30, 30, 41, 31))
        self.btn_setting.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("pyqt_icon/settings.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_setting.setIcon(icon1)
        self.btn_setting.setIconSize(QtCore.QSize(22, 22))
        self.btn_setting.setObjectName("btn_setting")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # self.btn_creat_area.clicked.connect(self.displayImage.clear) # type: ignore
        # self.btn_submit.clicked.connect(self.displayImage.clear) # type: ignore
        # self.btn_cancle.clicked.connect(self.displayImage.clear) # type: ignore
        # self.btn_Start.clicked.connect(self.displayImage.clear) # type: ignore
        # self.btn_resetArea.clicked.connect(self.displayImage.clear) # type: ignore
        # self.btn_open.clicked.connect(self.displayImage.clear) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "Result"))
        self.btn_Start.setText(_translate("MainWindow", "Start"))
        self.title.setText(_translate("MainWindow", "Wrong way detection from traffic video"))
        self.btn_submit.setText(_translate("MainWindow", "submit"))
        self.btn_cancle.setText(_translate("MainWindow", "cancel"))
        self.btn_stop.setText(_translate("MainWindow", "Stop"))
        self.btn_creat_area.setText(_translate("MainWindow", "Create Area"))
        self.label_fielname.setText(_translate("MainWindow", "import video"))
        self.btn_resetArea.setText(_translate("MainWindow", "reset"))
        self.btn_undo.setText(_translate("MainWindow", "undo"))
