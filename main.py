from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from win.qtUI import Ui_MainWindow
from win.setting import Ui_setting_Window
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtWidgets

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import cv2
import platform
import shutil
from pathlib import Path
import math
import os
import sys


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.insert(0, './yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort




class MainWindow(QMainWindow, Ui_MainWindow,Ui_setting_Window):
    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.settings_window = QtWidgets.QMainWindow()
        self.settings_ui = Ui_setting_Window()
        self.settings_ui.setupUi(self.settings_window)

        self.source = '0'
        self.area = []
        self.iou_thres = 0.28
        self.conf_thres = 0.17
        self.yolo_model = 'yolo_model/yolov5n.pt'
        self.classes = [2,3,5,7]
        self.filename_temp = ''
        self.filename_label_temp = ''

        self.iou_thres_default = 0.28
        self.conf_thres_default = 0.17
        self.yolo_model_default = 'yolo_model/yolov5n.pt'
        self.output_path = 'runs/track'

        self.btn_setting.clicked.connect(self.open_setting_window)
        self.btn_open.clicked.connect(self.open_file)
        self.btn_creat_area.clicked.connect(self.whileloop_rectangle)
        self.btn_creat_area.setEnabled(False)

        self.btn_submit.setEnabled(False)
        self.btn_cancle.setEnabled(False)
        self.btn_resetArea.setEnabled(False)
        self.btn_submit.setCheckable(True)
        self.btn_submit.setChecked(False)
        self.btn_cancle.setCheckable(True)
        self.btn_cancle.setChecked(False)
        self.btn_resetArea.clicked.connect(self.reset_area)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setCheckable(True)
        self.btn_undo.setChecked(False)
        self.btn_undo.clicked.connect(self.undo_retangle)

        self.btn_Start.clicked.connect(self.detect_video)
        self.btn_Start.setEnabled(False)
        self.btn_stop.hide()
        self.btn_stop.setCheckable(True)
        self.btn_stop.setChecked(False)

        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)

        self.settings_ui.checkBox_save_vid.setChecked(True)
        self.settings_ui.checkBox_save_txt.setChecked(True)
        self.settings_ui.checkBox_save_vid.toggled.connect(self.check_save_video)
        self.settings_ui.checkBox_save_txt.toggled.connect(self.check_save_txt)
        self.settings_ui.btn_setting_cancle.clicked.connect(self.cancle_settings)
        self.settings_ui.btn_setting_submit.clicked.connect(self.submit_settings)
        self.settings_ui.btn_setting_reset.clicked.connect(self.setting_reset)
        self.settings_ui.btn_setting_folder.clicked.connect(self.select_output_folder)
        self.settings_ui.label_setting_output_path.setText(self.output_path)

        self.settings_ui.checkBox_person.setChecked(False)
        self.settings_ui.checkBox_vehicle.setChecked(True)
        self.settings_ui.checkBox_person.toggled.connect(self.check_person)
        self.settings_ui.checkBox_vehicle.toggled.connect(self.check_vehicle)



        model_folder = 'yolo_model'
        files_model = os.listdir(model_folder)
        for file in files_model:
            self.settings_ui.setting_comboBox.addItem(file)
        self.settings_ui.setting_comboBox.setCurrentText('yolov5n.pt')
        self.settings_ui.setting_comboBox.currentIndexChanged.connect(self.select_model)
        # self.deep_sort_model = 'osnet_x0_25'


        self.deep_sort_model = 'osnet_ibn_x1_0'
        self.output = 'inference/output'
        self.imgsz = (640, 640)
        self.device = ''
        self.show_vid = True
        self.save_vid = True
        self.save_txt = True
        self.agnostic_nms = False
        self.augment = False
        self.evaluate = False
        self.config_deepsort = 'deep_sort/configs/deep_sort.yaml'
        self.half = False
        self.visualize = False
        self.max_det = 1000
        self.dnn = False
        self.project = 'runs/track'
        self.name = 'exp'
        self.exist_ok = False

        self.FILE = Path(__file__).resolve()
        self.ROOT = self.FILE.parents[0]  # yolov5 deepsort root directory
        if str(self.ROOT) not in sys.path:
            sys.path.append(str(self.ROOT))  # add ROOT to PATH
        self.OOT = Path(os.path.relpath(self.ROOT, Path.cwd()))


        print('classes',self.classes)


    def detect(self):
        self.area_list = []
        self.data_in = []
        self.data_out = []
        self.area_names = []
        self.area_result ={}

        for i,area in enumerate(self.area):
            self.area_list.append([])
            self.data_in.append([])
            self.data_out.append([])
            self.area_names.append(f'area{i+1}')

        for i in self.area_list:
            i.append([])
            i.append([])
            i.append(0)
            i.append(0)

        if 0 not in self.classes:
            self.area_result = {a: {
                'car_RW': [],
                'car_WW': [],
                'motorcycle_RW': [],
                'motorcycle_WW': [],
                'bus_RW': [],
                'bus_WW': [],
                'truck_RW': [],
                'truck_WW': []
            } for a in self.area_names}
        elif 0 in self.classes and len(self.classes) == 1:
            self.area_result = {a: {
                'person_RW': [],
                'person_WW': [],
            } for a in self.area_names}
        elif 0 in self.classes and len(self.classes) > 1:
            self.area_result = {a: {
                'person_RW': [],
                'person_WW': [],
                'car_RW': [],
                'car_WW': [],
                'motorcycle_RW': [],
                'motorcycle_WW': [],
                'bus_RW': [],
                'bus_WW': [],
                'truck_RW': [],
                'truck_WW': []
            } for a in self.area_names}

        source = str(self.source)
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        deepsort = DeepSort(self.deep_sort_model,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize

        device = select_device(self.device)
        self.half &= device.type != 'cpu'  # half precision only supported on CUDA

        # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
        # its own .txt file. Hence, in that case, the output folder is not restored
        if not self.evaluate:
            if os.path.exists(self.output):
                pass
                shutil.rmtree(self.output)  # delete output folder
            os.makedirs(self.output)  # make new output folder

        # Directories

        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # print('save_dir',save_dir)

        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.yolo_model, device=device, dnn=self.dnn)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if self.half else model.model.float()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment supports image displays
        if self.show_vid:
            show_vid = check_imshow()

        # Dataloader
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

        # print('txt_file_name', txt_file_name)
        # print('txt_path',txt_path)

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1


            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = model(img, augment=self.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                       self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string



                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h = im0.shape[1], im0.shape[0]
                for i ,area in enumerate(self.area):
                    self.draw_rectangle(im0,area,i+1)



                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # self.listWidget.addItem(f"{n} {names[int(c)]}{'s' * (n > 1)}")
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class
                            # label = f'{id} {names[c]} {conf:.2f}'
                            label = f'{id} '

                            annotator.box_label(bboxes, label, color=colors(c, True))

                            self.counting(bboxes,id,names[c])


                    self.update_progress_bar(frame_idx+1)
                    # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

                else:
                    deepsort.increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()



                if self.show_vid:
                    self.set_display_iamge(im0)
                    # cv2.imshow(str(p), im0)
                    elapsed_time = cv2.getTickCount() - self.start_time
                    self.wait_time = max(self.wait_time - int(elapsed_time), 1)
                    key = cv2.waitKey(self.wait_time) & 0xFF
                    if key == ord('q'):  # Press 'Esc' to exit
                        break
                        self.btn_Start.show()
                        self.btn_stop.hide()
                        self.btn_creat_area.setEnabled(True)
                        self.btn_open.setEnabled(True)
                        self.btn_stop.setChecked(True)
                        self.progressBar.setValue(0)
                        if len(self.area) != 0:
                            self.btn_resetArea.setEnabled(True)
                            self.btn_undo.setEnabled(True)
                        raise StopIteration
                        cv2.destroyAllWindows()
                    self.start_time = cv2.getTickCount()

                # Save results (image with detections)
                if self.save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            if self.btn_stop.isChecked() :
                self.btn_Start.show()
                self.btn_stop.hide()
                self.btn_creat_area.setEnabled(True)
                self.btn_open.setEnabled(True)
                self.btn_stop.setChecked(False)
                self.progressBar.setValue(0)
                if len(self.area) != 0:
                    self.btn_resetArea.setEnabled(True)
                    self.btn_undo.setEnabled(True)
                break
                cv2.destroyAllWindows()

            self.listWidget.clear()
            if self.area != []:
                for area, value in self.area_result.items():
                    if 0 not in self.classes:
                        car_RW_count = len(value['car_RW'])
                        car_WW_count = len(value['car_WW'])
                        motorcycle_RW_count = len(value['motorcycle_RW'])
                        motorcycle_WW_count = len(value['motorcycle_WW'])
                        bus_RW_count = len(value['bus_RW'])
                        bus_WW_count = len(value['bus_WW'])
                        truck_RW_count = len(value['truck_RW'])
                        truck_WW_count = len(value['truck_WW'])
                        total_RW_count = car_RW_count + motorcycle_RW_count + bus_RW_count + truck_RW_count
                        total_WW_count = car_WW_count + motorcycle_WW_count + bus_WW_count + truck_WW_count
                        self.listWidget.addItem(f'------- {area}--------')
                        self.listWidget.addItem(
                            f'motorcycle - right {motorcycle_RW_count} : wrong {motorcycle_WW_count}')
                        self.listWidget.addItem(f'car - right {car_RW_count} : wrong {car_WW_count}')
                        self.listWidget.addItem(f'Bus - right {bus_RW_count} : wrong {bus_WW_count}')
                        self.listWidget.addItem(f'Truck - right {truck_RW_count} : wrong {truck_WW_count}')
                        self.listWidget.addItem(f'Total - right {total_RW_count} : wrong {total_WW_count}')
                    elif 0 in self.classes and len(self.classes) == 1:
                        person_RW_count = len(value['person_RW'])
                        person_WW_count = len(value['person_WW'])
                        self.listWidget.addItem(f'------- {area}--------')
                        self.listWidget.addItem(
                            f'person -right {person_RW_count} : wrong {person_WW_count}')
                        self.listWidget.addItem(f'Total - right {person_RW_count} : wrong {person_WW_count}')
                    elif 0 in self.classes and len(self.classes) > 1:
                        person_RW_count = len(value['person_RW'])
                        person_WW_count = len(value['person_WW'])
                        car_RW_count = len(value['car_RW'])
                        car_WW_count = len(value['car_WW'])
                        motorcycle_RW_count = len(value['motorcycle_RW'])
                        motorcycle_WW_count = len(value['motorcycle_WW'])
                        bus_RW_count = len(value['bus_RW'])
                        bus_WW_count = len(value['bus_WW'])
                        truck_RW_count = len(value['truck_RW'])
                        truck_WW_count = len(value['truck_WW'])
                        total_RW_count = person_RW_count + car_RW_count + motorcycle_RW_count + bus_RW_count + truck_RW_count
                        total_WW_count = person_WW_count + car_WW_count + motorcycle_WW_count + bus_WW_count + truck_WW_count
                        self.listWidget.addItem(f'------- {area}--------')
                        self.listWidget.addItem(
                            f'person - right {person_RW_count} : wrong {person_WW_count}')
                        self.listWidget.addItem(
                            f'motorcycle - right {motorcycle_RW_count} : wrong {motorcycle_WW_count}')
                        self.listWidget.addItem(f'car - right {car_RW_count} : wrong {car_WW_count}')
                        self.listWidget.addItem(f'Bus - right {bus_RW_count} : wrong {bus_WW_count}')
                        self.listWidget.addItem(f'Truck - right {truck_RW_count} : wrong {truck_WW_count}')
                        self.listWidget.addItem(f'Total - right {total_RW_count} : wrong {total_WW_count}')


        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
               per image at shape {(1, 3, *imgsz)}' % t)

        if self.save_txt:
            with open(txt_path, 'a') as f:
                if self.area != []:
                    for area, value in self.area_result.items():
                        if 0 not in self.classes:
                            car_RW_count = len(value['car_RW'])
                            car_WW_count = len(value['car_WW'])
                            motorcycle_RW_count = len(value['motorcycle_RW'])
                            motorcycle_WW_count = len(value['motorcycle_WW'])
                            bus_RW_count = len(value['bus_RW'])
                            bus_WW_count = len(value['bus_WW'])
                            truck_RW_count = len(value['truck_RW'])
                            truck_WW_count = len(value['truck_WW'])
                            total_RW_count = car_RW_count + motorcycle_RW_count + bus_RW_count + truck_RW_count
                            total_WW_count = car_WW_count + motorcycle_WW_count + bus_WW_count + truck_WW_count

                            f.write(f'------- {area}--------\n')
                            f.write(f'motorcycle - right {motorcycle_RW_count} : wrong {motorcycle_WW_count}\n')
                            f.write(f'car        - right {car_RW_count} : wrong {car_WW_count}\n')
                            f.write(f'Bus        - right {bus_RW_count} : wrong {bus_WW_count}\n')
                            f.write(f'Truck      - right {truck_RW_count} : wrong {truck_WW_count}\n')
                            f.write(f'Total      - right {total_RW_count} : wrong {total_WW_count}\n')
                        elif 0 in self.classes and len(self.classes) == 1:
                            person_RW_count = len(value['person_RW'])
                            person_WW_count = len(value['person_WW'])
                            f.write(f'------- {area}--------\n')
                            f.write(f'person - right {person_RW_count} : wrong {person_WW_count}\n')
                            f.write(f'Total  - right {person_RW_count} : wrong {person_WW_count}\n')


                        elif 0 in self.classes and len(self.classes) > 1:
                            person_RW_count = len(value['person_RW'])
                            person_WW_count = len(value['person_WW'])
                            car_RW_count = len(value['car_RW'])
                            car_WW_count = len(value['car_WW'])
                            motorcycle_RW_count = len(value['motorcycle_RW'])
                            motorcycle_WW_count = len(value['motorcycle_WW'])
                            bus_RW_count = len(value['bus_RW'])
                            bus_WW_count = len(value['bus_WW'])
                            truck_RW_count = len(value['truck_RW'])
                            truck_WW_count = len(value['truck_WW'])
                            total_RW_count = person_RW_count + car_RW_count + motorcycle_RW_count + bus_RW_count + truck_RW_count
                            total_WW_count = person_WW_count + car_WW_count + motorcycle_WW_count + bus_WW_count + truck_WW_count

                            f.write(f'------- {area}--------\n')
                            f.write(f'person     - right {person_RW_count} : wrong {person_WW_count}\n')
                            f.write(f'motorcycle - right {motorcycle_RW_count} : wrong {motorcycle_WW_count}\n')
                            f.write(f'car        - right {car_RW_count} : wrong {car_WW_count}\n')
                            f.write(f'Bus        - right {bus_RW_count} : wrong {bus_WW_count}\n')
                            f.write(f'Truck      - right {truck_RW_count} : wrong {truck_WW_count}\n')
                            f.write(f'Total      - right {total_RW_count} : wrong {total_WW_count}\n')



        if self.save_txt or self.save_vid:
            print('Results saved to %s' % save_path)
            QMessageBox.question(self, 'Results saved', 'Results saved to %s' % save_path)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

        self.btn_Start.show()
        self.btn_stop.hide()
        self.btn_creat_area.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.progressBar.setValue(0)
        if len(self.area) != 0:
            self.btn_resetArea.setEnabled(True)
            self.btn_undo.setEnabled(True)
        cv2.destroyAllWindows()

    def counting(self,box,id,name):
        center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))
        # object_x = center_coordinates[0]
        # object_y = center_coordinates[1]
        if len(self.area) != 0:
            for i,area in enumerate(self.area):
                inout = self.aera_line_position(area)
                index = i
                for i, inAndOut in enumerate(inout):
                    pts = inAndOut
                    pts_array = np.array(pts, np.int32)
                    dist = cv2.pointPolygonTest(pts_array, center_coordinates, False)
                    if i == 1:
                        # line in
                        if dist == 1:
                            # Object Center is inside the rectangle
                            if id in self.data_out[index] and id not in self.area_list[index][1]:
                                self.area_list[index][1].append(id)
                                self.area_list[index][3] += 1
                                if name == 'person':
                                    self.area_result[f'area{index + 1}']['person_WW'].append(id)
                                elif name == 'car':
                                    self.area_result[f'area{index + 1}']['car_WW'].append(id)
                                elif name == 'motorcycle':
                                    self.area_result[f'area{index + 1}']['motorcycle_WW'].append(id)
                                elif name == 'bus':
                                    self.area_result[f'area{index + 1}']['bus_WW'].append(id)
                                elif name == 'truck':
                                    self.area_result[f'area{index + 1}']['truck_WW'].append(id)
                            if id not in self.data_in[index]:
                                self.data_in[index].append(id)
                    else:
                        # line out
                        if dist == 1:
                            # Object Center is inside the rectangle
                            if id in self.data_in[index] and id not in self.area_list[index][0]:
                                self.area_list[index][0].append(id)
                                self.area_list[index][2] += 1
                                if name == 'person':
                                    self.area_result[f'area{index + 1}']['person_RW'].append(id)
                                elif name == 'car':
                                    self.area_result[f'area{index + 1}']['car_RW'].append(id)
                                elif name == 'motorcycle':
                                    self.area_result[f'area{index + 1}']['motorcycle_RW'].append(id)
                                elif name == 'bus':
                                    self.area_result[f'area{index + 1}']['bus_RW'].append(id)
                                elif name == 'truck':
                                    self.area_result[f'area{index + 1}']['truck_RW'].append(id)
                            if id not in self.data_out[index]:
                                self.data_out[index].append(id)


    def select_model(self):
        self.yolo_model = f'yolo_model/{self.settings_ui.setting_comboBox.currentText()}'

    def check_save_video(self):
        if self.settings_ui.checkBox_save_vid.isChecked():
            self.save_vid = True
        else:
            self.save_vid = False
    def check_save_txt(self):
        if self.settings_ui.checkBox_save_txt.isChecked():
            self.save_txt = True
        else:
            self.save_txt = False

    def check_person(self):
        if self.settings_ui.checkBox_person.isChecked():
            self.classes.append(0)
            self.classes.sort()
        else:
            self.classes.remove(0)

    def check_vehicle(self):
        if self.settings_ui.checkBox_vehicle.isChecked():
            self.classes.extend([2,3,5,7])
            self.classes.sort()
        else:
            self.remove_vehicle = [2,3,5,7]
            for remove in self.remove_vehicle:
                if remove in self.classes:
                    self.classes.remove(remove)


    def update_iou_slider(self, value):
        iou = round(value/100, 2)
        self.settings_ui.slider_iou_setting.setValue(value)
        self.settings_ui.SpinBox_iou_setting.setValue(iou)
        self.iou_thres_temp = iou

    def update_iou_spinBox(self, value):
        iou = int(value*100)
        self.settings_ui.slider_iou_setting.setValue(iou)
        self.settings_ui.SpinBox_iou_setting.setValue(value)
        self.iou_thres_temp = value

    def update_conf_slider(self, value):
        conf = round(value / 100, 2)
        self.settings_ui.slider_conf_setting.setValue(value)
        self.settings_ui.SpinBox_conf_setting.setValue(conf)
        self.conf_thres_temp = conf

    def update_conf_spinBox(self, value):
        conf = int(value * 100)
        self.settings_ui.slider_conf_setting.setValue(conf)
        self.settings_ui.SpinBox_conf_setting.setValue(value)
        self.conf_thres_temp = value

    def open_file(self):
        if len(self.area) != 0:
            self.reset_area()
        self.filename= ''
        try:
            self.filename = QFileDialog.getOpenFileName(self, 'Open Video')
            self.filename_label = os.path.basename(self.filename[0])
        except:
            pass

        if self.filename == ('',''):
            self.source = self.filename_temp
            # self.label_fielname.setText(self.filename_label_temp)
        else:
            self.filename_temp = self.filename[0]
            self.filename_label_temp = os.path.basename(self.filename[0])
            self.source = self.filename[0]
            # self.label_fielname.setText(self.filename_label)
        self.progressBar.setValue(0)

        self.rectangle_color = (0, 255, 0)
        self.circle_color = (0, 0, 255)
        self.line_thickness = 2
        self.circle_radius = 6

        self.line_color = (0, 0, 255)
        self.circle_line_color = (255, 0, 0)

        video = cv2.VideoCapture(self.source)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(0, total_frames - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = video.read()
        fps = video.get(cv2.CAP_PROP_FPS)
        self.start_time = cv2.getTickCount()
        self.wait_time = int(1000 / 60)

        # frame = self.resize_bg(frame)

        self.background = frame.copy()
        self.background_next = frame
        self.background_orginal = frame.copy()

        self.progressBar.setMaximum(total_frames)
        self.set_display_iamge(self.background)
        self.center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))

        # Global variables
        self.circles = [(self.center[0] - 150, self.center[1] - 150), (self.center[0] + 150, self.center[1] - 150),
                        (self.center[0] + 150, self.center[1] + 150), (self.center[0] - 150, self.center[1] + 150)]
        self.circles_line = [(self.center[0], self.center[1] - 150), (self.center[0], self.center[1] + 150)]

        self.selected_circle_index = None
        self.previous_mouse_position = None
        # self.draw_rectangle(self.background,self.circles)

        self.btn_creat_area.setEnabled(True)
        self.btn_Start.setEnabled(True)

    def aera_line_position(self,area):
        inout = []
        for i in range(2):
            if i == 0:
                line = (area[0],area[1])

            else:
                line = (area[3],area[2])
            # Calculate the length of the line
            length = math.sqrt((line[1][0] - line[0][0]) ** 2 + (line[1][1] - line[0][1]) ** 2)
            # Calculate the angle of the line
            angle = math.degrees(math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0]))
            # Define the rectangle parameters
            rect_center = ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)
            rect_size = (int(length), 25)
            rect_angle = angle

            # Calculate the four corner points of the rectangle
            rect = (rect_center, rect_size, rect_angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            inout.append(box)

        return inout

    def draw_rectangle(self,image,circles,index=False):
        """Draws a rectangle with circles on its corners and lines connecting them"""
        for i in range(4):
            cv2.circle(image, tuple(circles[i]), self.circle_radius, self.circle_color, -1)
            cv2.line(image, tuple(circles[i]), tuple(circles[(i + 1) % 4]), self.rectangle_color, self.line_thickness)

        polyline = False

        if polyline == True:
            for i in range(2):
                if i == 0:
                    line = (circles[0],circles[1])

                else:
                    line = (circles[3],circles[2])
                # calculate the length of the line
                # Calculate the length of the line
                length = math.sqrt((line[1][0] - line[0][0]) ** 2 + (line[1][1] - line[0][1]) ** 2)

                # Calculate the angle of the line
                angle = math.degrees(math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0]))

                # Define the rectangle parameters
                rect_center = ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)
                rect_size = (int(length), 25)
                rect_angle = angle

                # Calculate the four corner points of the rectangle
                rect = (rect_center, rect_size, rect_angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

        arrow_start = (int((circles[2][0] - circles[3][0]) / 2), int((circles[2][1] + circles[3][1]) / 2))
        arrow_start = (circles[3][0] + arrow_start[0], arrow_start[1])
        arrow_end = (int((circles[1][0] - circles[0][0]) / 2), int((circles[1][1] + circles[0][1]) / 2))
        arrow_end = (circles[0][0] + arrow_end[0], arrow_end[1])
        cv2.arrowedLine(image, arrow_start, arrow_end, (0, 255, 0), self.line_thickness, tipLength=0.12)

        center_x = int(np.mean([point[0] for point in circles]))
        center_y = int(np.mean([point[1] for point in circles]))

        if index == False :
            id = len(self.area)+1
            text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        elif index != False:
            id = index
            text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]

        background_width = text_size[0] + 20
        background_height = text_size[1] + 10
        background_points = [(center_x - int(background_width / 2), center_y - int(background_height / 2)),
                             (center_x + int(background_width / 2), center_y - int(background_height / 2)),
                             (center_x + int(background_width / 2), center_y + int(background_height / 2)),
                             (center_x - int(background_width / 2), center_y + int(background_height / 2))]

        cv2.rectangle(image, background_points[0], background_points[2], self.rectangle_color, cv2.FILLED)

        cv2.putText(image, str(id), (center_x - int(text_size[0] / 2), center_y + int(text_size[1] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def draw_line(self,image, circles):
        """Draws a line between the first and last circles in the list"""
        cv2.circle(image, tuple(circles[0]), self.circle_radius, self.circle_line_color, -1)
        cv2.circle(image, tuple(circles[-1]), self.circle_radius, self.circle_line_color, -1)
        cv2.line(image, tuple(circles[0]), tuple(circles[-1]), self.line_color, self.line_thickness)

    def find_closest_circle_index(self,point, circles):
        """Finds the index of the closest circle to a given point"""
        distances = [(i, np.linalg.norm(np.array(point) - np.array(circle))) for i, circle in enumerate(circles)]
        closest_circle_index = min(distances, key=lambda d: d[1])[0]
        return closest_circle_index

    def mouse_callback(self,event, x, y,flags, param):
        """Handles mouse events and updates the selected circle"""
        global selected_circle_index, previous_mouse_position
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_circle_index = self.find_closest_circle_index((x, y), self.circles)
            self.previous_mouse_position = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_circle_index is not None:
            dx, dy = x - self.previous_mouse_position[0], y - self.previous_mouse_position[1]
            self.circles[self.selected_circle_index] = (
                self.circles[self.selected_circle_index][0] + dx, self.circles[self.selected_circle_index][1] + dy)
            self.previous_mouse_position = (x, y)

            # Remove previous rectangle by drawing background over it
            self.background[:] = self.background_next
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_circle_index = None

    def undo_retangle(self):
        if self.area != []:
            self.area.pop()
            self.background = self.background_orginal
            self.background_next = self.background_orginal
            self.background_orginal = self.background_orginal.copy()

            for i, area in enumerate(self.area):
                self.draw_rectangle(self.background, area, i + 1)
            self.set_display_iamge(self.background)
            if self.area == []:
                self.btn_undo.setEnabled(False)
                self.btn_resetArea.setEnabled(False)

    def whileloop_rectangle(self):
        if self.filename == '':
            pass
        else:
            self.btn_submit.setEnabled(True)
            self.btn_cancle.setEnabled(True)
            self.btn_resetArea.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_Start.setEnabled(False)

            cv2.namedWindow('rectangle')
            cv2.setMouseCallback('rectangle', self.mouse_callback)
            self.btn_creat_area.setEnabled(False)
            screen_width, screen_height = cv2.getWindowImageRect('rectangle')[2:]
            window_width, window_height = cv2.getWindowImageRect('rectangle')[2:]
            center_x = int((screen_width - window_width) / 2)
            center_y = int((screen_height - window_height) / 2)
            cv2.moveWindow('rectangle', center_x, center_y)

            if len(self.area) == 0:
                while True:
                    image = self.background.copy()
                    self.draw_rectangle(image, self.circles)
                    cv2.imshow('rectangle', image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or self.btn_submit.isChecked() or self.btn_cancle.isChecked():
                        break
            else:
                while True:
                    image = self.background_next.copy()
                    self.draw_rectangle(image, self.circles)
                    cv2.imshow('rectangle', image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or self.btn_submit.isChecked() or self.btn_cancle.isChecked():
                        break

            cv2.destroyAllWindows()
            if (self.btn_submit.isChecked() == True):
                self.area.append(self.circles)
                self.draw_rectangle(self.background_next, self.circles,len(self.area))

                self.circles = [(self.center[0] - 150, self.center[1] - 150),
                                (self.center[0] + 150, self.center[1] - 150),
                                (self.center[0] + 150, self.center[1] + 150),
                                (self.center[0] - 150, self.center[1] + 150)]
                self.selected_circle_index = None
                self.previous_mouse_position = None
                self.set_display_iamge(self.background_next)
                self.btn_resetArea.setEnabled(True)
                self.btn_undo.setEnabled(True)
                self.btn_creat_area.setEnabled(True)
                self.btn_Start.setEnabled(True)
            if (self.btn_cancle.isChecked() == True):
                self.circles = []
                self.set_display_iamge(self.background_next)
                self.btn_creat_area.setEnabled(True)
                self.btn_undo.setEnabled(False)
                self.btn_Start.setEnabled(True)
                self.btn_resetArea.setEnabled(True)
                self.btn_undo.setEnabled(True)
                self.selected_circle_index = None
                self.previous_mouse_position = None
                self.circles = [(self.center[0] - 150, self.center[1] - 150),
                                (self.center[0] + 150, self.center[1] - 150),
                                (self.center[0] + 150, self.center[1] + 150),
                                (self.center[0] - 150, self.center[1] + 150)]
            self.btn_submit.setChecked(False)
            self.btn_cancle.setChecked(False)
            self.btn_submit.setEnabled(False)
            self.btn_cancle.setEnabled(False)
            self.btn_Start.setEnabled(True)

    def resize_bg(self,image):
        ih, iw, _ = image.shape
        w = 960
        h = 540

        if iw / w > ih / h:
            scal = w / iw
            nw = w
            nh = int(scal * ih)
            image = cv2.resize(image, (nw, nh))

        else:
            scal = h / ih
            nw = int(scal * iw)
            nh = h
            image = cv2.resize(image, (nw, nh))

        return image

    def set_display_iamge(self,image):
        ih, iw, _ = image.shape
        w = self.displayImage.geometry().width()
        h = self.displayImage.geometry().height()
        if iw / w > ih / h:
            scal = w / iw
            nw = w
            nh = int(scal * ih)
            image = cv2.resize(image, (nw, nh))

        else:
            scal = h / ih
            nw = int(scal * iw)
            nh = h
            image = cv2.resize(image, (nw, nh))


        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        # self.displayImage.setPixmap(QtGui.QPixmap.fromImage(image))
        self.displayImage.setPixmap(QPixmap.fromImage(image))

    def resize_iamge(self,image):
        # print(image.shape)
        ih, iw, = image.shape[0],image.shape[1]
        w = 1280
        h = 720
        if iw / w > ih / h:
            scal = w / iw
            nw = w
            nh = int(scal * ih)
            image = cv2.resize(image, (nw, nh))

        else:
            scal = h / ih
            nw = int(scal * iw)
            nh = h
            image = cv2.resize(image, (nw, nh))

        return image

    def reset_area(self):
        reply = QMessageBox.question(self, 'Reset Area', 'Are you sure you want to Reset Area?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.area = []
            self.background = self.background_orginal
            self.background_next = self.background_orginal
            self.background_orginal = self.background_orginal.copy()

            self.set_display_iamge(self.background)
            self.btn_resetArea.setEnabled(False)
            self.btn_creat_area.setEnabled(True)
            self.btn_undo.setEnabled(False)

            self.selected_circle_index = None
            self.previous_mouse_position = None
        else:
            pass

    def detect_video(self):
        # print('area',self.area)
        self.btn_Start.hide()
        self.btn_stop.show()
        self.btn_submit.setEnabled(False)
        self.btn_cancle.setEnabled(False)
        self.btn_creat_area.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_resetArea.setEnabled(False)
        self.btn_undo.setEnabled(False)
        with torch.no_grad():
            self.detect()

    def update_progress_bar(self,frame):
        self.progressBar.setValue(frame)

    def open_setting_window(self):
        self.settings_window.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.settings_window.show()
        self.set_value()

        self.iou_thres_temp = self.iou_thres
        self.conf_thres_temp = self.conf_thres
        self.yolo_model_temp = self.yolo_model

    def set_value(self):
        self.settings_ui.SpinBox_iou_setting.setMinimum(0.00)
        self.settings_ui.SpinBox_iou_setting.setMaximum(1.00)
        self.settings_ui.SpinBox_iou_setting.setValue(self.iou_thres)
        self.settings_ui.SpinBox_iou_setting.setSingleStep(0.01)
        self.settings_ui.slider_iou_setting.setMinimum(0)
        self.settings_ui.slider_iou_setting.setMaximum(100)
        self.settings_ui.slider_iou_setting.setValue(int(self.iou_thres * 100))
        self.settings_ui.SpinBox_iou_setting.valueChanged.connect(self.update_iou_spinBox)
        self.settings_ui.slider_iou_setting.valueChanged.connect(self.update_iou_slider)
        self.settings_ui.SpinBox_conf_setting.setMinimum(0.00)
        self.settings_ui.SpinBox_conf_setting.setMaximum(1.00)
        self.settings_ui.SpinBox_conf_setting.setValue(self.conf_thres)
        self.settings_ui.SpinBox_conf_setting.setSingleStep(0.01)
        self.settings_ui.slider_conf_setting.setMinimum(0)
        self.settings_ui.slider_conf_setting.setMaximum(100)
        self.settings_ui.slider_conf_setting.setValue(int(self.conf_thres * 100))
        self.settings_ui.slider_conf_setting.valueChanged.connect(self.update_conf_slider)
        self.settings_ui.SpinBox_conf_setting.valueChanged.connect(self.update_conf_spinBox)

    def cancle_settings(self):
        self.iou_thres_temp = self.iou_thres
        self.conf_thres_temp = self.conf_thres
        self.yolo_model_temp = self.yolo_model

        self.settings_ui.SpinBox_iou_setting.setValue(self.iou_thres)
        self.settings_ui.slider_iou_setting.setValue(int(self.iou_thres * 100))

        self.settings_ui.SpinBox_conf_setting.setValue(self.conf_thres)
        self.settings_ui.slider_conf_setting.setValue(int(self.conf_thres * 100))

        self.settings_window.hide()

    def submit_settings(self):
        self.iou_thres = self.iou_thres_temp
        self.conf_thres = self.conf_thres_temp
        self.yolo_model = self.yolo_model_temp

        self.settings_window.hide()
        self.select_model()
        self.project = self.output_path
        # print(self.yolo_model)
        # print('save_vid',self.save_vid)
        # print('save_txt',self.save_txt)
        # print('output_path', self.output_path)
        # print('classes',self.classes)


    def setting_reset(self):
        self.iou_thres = self.iou_thres_default
        self.conf_thres = self.conf_thres_default
        self.yolo_model = self.yolo_model_default

        self.settings_ui.SpinBox_iou_setting.setValue(self.iou_thres)
        self.settings_ui.slider_iou_setting.setValue(int(self.iou_thres * 100))

        self.settings_ui.SpinBox_conf_setting.setValue(self.conf_thres)
        self.settings_ui.slider_conf_setting.setValue(int(self.conf_thres * 100))

        self.settings_ui.setting_comboBox.setCurrentText('yolov5n.pt')

        self.settings_ui.checkBox_save_vid.setChecked(True)
        self.settings_ui.checkBox_save_txt.setChecked(True)

    def select_output_folder(self):
        self.output_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.settings_ui.label_setting_output_path.setText(self.output_path)

    def closeEvent(self, event):
        # Prompt the user before closing the window
        reply = QMessageBox.question(self, 'Close Window', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



def main():
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()