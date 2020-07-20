import sys
import numpy as np
import math

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from cv2 import cv2

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FormatStrFormatter
from matplotlib.figure import Figure

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Reversible Image Watermarking Based on Modified Histogram Shifting Technique'
        self.left = 40
        self.top = 60

        self.width = 1800
        self.height = 900
        self.width_img = 16*60
        self.height_img = 9*60
        self.width_watermark = 64
        self.height_watermark = 64

        self.cv_img = None
        self.label_imgOrig = QLabel()

        self.cv_imgShift = None
        self.label_imgShift = QLabel()

        self.cv_imgDiff = None
        self.label_imgDiff = QLabel()

        self.cv_imgEmbed = None
        self.label_imgEmbed = QLabel()

        self.cv_watermarkEmb = None
        self.label_watermarkEmb = QLabel()

        self.cv_watermarkExt = None
        self.label_watermarkExt = QLabel()

        self.cv_imgRestored = None
        self.label_imgRestored = QLabel()

        self.idx_max = 0
        self.idx_min = 0
        self.embedPoint = 0
        self.savePoints = set()

        self.flag_loadedImage = False
        self.flag_loadedWatermark = False
        
        self.figure_histSection = Figure(figsize=(5, 4))
        self.canvas_histSection = FigureCanvas(self.figure_histSection)
        self.toolbar_histSection = NavigationToolbar(self.canvas_histSection, self)

        self.figure_histWatermark = Figure(figsize=(5, 6))
        self.canvas_histWatermark = FigureCanvas(self.figure_histWatermark)
        self.toolbar_histWatermark = NavigationToolbar(self.canvas_histWatermark, self)

        self.initUI()

    def showImage(self, cv_img, w, h, lbl):
        img = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.strides[0], QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio)
        lbl.setPixmap(pixmap)
        
    def restoreImage(self):
        be = min(self.idx_min, self.idx_max)
        en = max(self.idx_min, self.idx_max)

        if self.idx_max < self.idx_min:     #Right shift
            be += 2
            en += 1
        else:
            en -= 2
            be -= 1

        ri, ci = self.cv_imgRestored.shape
        for i in range(ri):
            for j in range(ci):
                cur = int(self.cv_imgRestored[i][j])
                
                if cur >= be and cur <= en:
                    if self.idx_max < self.idx_min:
                        self.cv_imgRestored[i][j] -= 1
                        if self.cv_imgRestored[i][j] == self.idx_min and (i,j) not in self.savePoints:
                            self.cv_imgRestored[i][j] += 1
                    else:
                        self.cv_imgRestored[i][j] += 1
                        if self.cv_imgRestored[i][j] == self.idx_min and (i,j) not in self.savePoints:
                            self.cv_imgRestored[i][j] -= 1
    
        self.showImage(self.cv_imgRestored, self.width_img/3, self.height_img/3, self.label_imgRestored)

    def extractWatermark(self):
        if self.flag_loadedImage == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Load Image First')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        if self.flag_loadedWatermark == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Load and Embed Watermark First')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        self.cv_watermarkExt = np.ndarray(shape=self.cv_watermarkEmb.shape, dtype=np.uint8)
        self.cv_imgRestored = np.copy(self.cv_imgEmbed)
        
        ri, ci = self.cv_imgRestored.shape
        positions = []
        for i in range(ri):
            for j in range(ci):
                if (self.cv_imgRestored[i][j]==self.idx_max) or (self.cv_imgRestored[i][j]==self.embedPoint):
                    positions.append((i,j))

        rw, cw = self.cv_watermarkExt.shape
        p = 0
        for i in range(rw):
            for j in range(cw):
                if self.cv_imgRestored[positions[p]] == self.embedPoint:
                    self.cv_watermarkExt[i][j] = 255
                    self.cv_imgRestored[positions[p]] = self.idx_max
                else:
                    self.cv_watermarkExt[i][j] = 0
                p += 1

        self.showImage(self.cv_watermarkExt, self.width_watermark*2, self.height_watermark*2, self.label_watermarkExt)  

        ### Set Restored Image Histogram ###
        self.restoreImage()   
        hist_restored = self.figure_histWatermark.add_subplot(2, 1, 2)
        hist_restored.clear()
        
        y, x, _ = hist_restored.hist(self.cv_imgRestored.ravel(), bins=255, range=[0,255], color='lightblue', align='left', edgecolor='black', linewidth=.1)
        hist_restored.title.set_text('Histogram of Restored Image, PSNR=' + str(psnr(self.cv_img, self.cv_imgRestored)))

        self.figure_histWatermark.tight_layout()
        self.canvas_histWatermark.draw()

    def embedWatermark(self):
        self.cv_imgEmbed = np.copy(self.cv_imgShift)

        ri, ci = self.cv_imgEmbed.shape
        positions = []
        for i in range(ri):
            for j in range(ci):
                if self.cv_imgEmbed[i][j] == self.idx_max:
                    positions.append((i,j))

        rw, cw = self.cv_watermarkEmb.shape
        p = 0
        for i in range(rw):
            for j in range(cw):
                if self.cv_watermarkEmb[i][j] == 255:
                    self.cv_imgEmbed[positions[p]] = self.embedPoint
                p += 1

        self.showImage(self.cv_imgEmbed, self.width_img/3, self.height_img/3, self.label_imgEmbed)

        ### Set Embed Histogram ###
        hist_embed = self.figure_histWatermark.add_subplot(2, 1, 1)
        hist_embed.clear()
        
        y, x, _ = hist_embed.hist(self.cv_imgEmbed.ravel(), bins=255, range=[0,255], color='lightblue', align='left', edgecolor='black', linewidth=.1)
        hist_embed.title.set_text('Histogram of Embedded Image, PSNR=' + str(psnr(self.cv_img, self.cv_imgEmbed)))

        _[self.embedPoint].set_fc('yellow')

        self.figure_histWatermark.tight_layout()
        self.canvas_histWatermark.draw()

    def getWatermark(self):
        if self.flag_loadedImage == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Load Image First')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        img_path, type = QFileDialog.getOpenFileName(self, 'Open Image', 'C:\\Users\\Antik\\Desktop\\code\\riwProject\\watermarks', 'Image Files (*.jpg *.jpeg *.png)')

        if len(img_path) == 0:
            return

        self.flag_loadedWatermark = True
    
        self.cv_watermarkEmb = cv2.imread(img_path, 2)
        self.cv_watermarkEmb = cv2.resize(self.cv_watermarkEmb,(self.width_watermark, self.height_watermark))
        ret, self.cv_watermarkEmb = cv2.threshold(self.cv_watermarkEmb,127,255,cv2.THRESH_BINARY)

        self.showImage(self.cv_watermarkEmb, self.width_watermark*2, self.height_watermark*2, self.label_watermarkEmb)
        self.embedWatermark()
    
    def shiftHistogram(self):
        self.cv_imgShift = np.copy(self.cv_img)
        self.cv_imgDiff = np.copy(self.cv_img)

        be = min(self.idx_min, self.idx_max)
        en = max(self.idx_min, self.idx_max)

        if self.idx_max < self.idx_min:
            be += 1
            self.embedPoint = be
        else:
            en -= 1
            self.embedPoint = en

        row, col = self.cv_imgShift.shape
        self.savePoints = set()
        for i in range(row):
            for j in range(col):
                cur = int(self.cv_imgShift[i][j])

                if cur == self.idx_min:
                    self.savePoints.add((i,j))
                
                if cur >= be and cur <= en:
                    if self.idx_max < self.idx_min:
                        self.cv_imgShift[i][j] += 1
                    else:
                        self.cv_imgShift[i][j] -= 1

                if self.cv_img[i][j] == self.cv_imgShift[i][j]:
                    self.cv_imgDiff[i][j] = 255
                else:
                    self.cv_imgDiff[i][j] = 0

        self.showImage(self.cv_imgShift, self.width_img/3, self.height_img/3, self.label_imgShift)
        self.showImage(self.cv_imgDiff, self.width_img/3, self.height_img/3, self.label_imgDiff)

    def setHistogramsSection(self):
        ### Original Histogram ###
        hist_orig = self.figure_histSection.add_subplot(3, 1, 1)
        hist_orig.clear()
        
        hist_orig.hist(self.cv_img.ravel(), bins=255, range=[0,255], color='lightblue', align='left', edgecolor='black', linewidth=.1)
        hist_orig.title.set_text('Histogram of Original Image')
        
        ### Highlighted Bins Histogram ###
        hist_high = self.figure_histSection.add_subplot(3, 1, 2)
        hist_high.clear()

        yHigh, xHigh, pHigh = hist_high.hist(self.cv_img.ravel(), bins=255, range=[0,255], color='lightblue', align='left', edgecolor='black', linewidth=.1)

        sortedCounts = np.sort(yHigh)
        self.idx_min = int(np.where(yHigh == sortedCounts[-2])[0][0])    #TODO change shift
        self.idx_max = int(np.where(yHigh == sortedCounts[-1])[0][0])   

        pHigh[self.idx_max].set_fc('r')
        pHigh[self.idx_min].set_fc('r')
        hist_high.title.set_text('\nIdentified Bins: Peak=' + str(self.idx_max) + ', Zero=' + str(self.idx_min))

        ### Shifted Bins Histogram ###
        self.shiftHistogram()

        hist_shift = self.figure_histSection.add_subplot(3, 1, 3)
        hist_shift.clear()
        
        hist_shift.hist(self.cv_imgShift.ravel(), bins=255, range=[0,255], color='lightblue', align='left', edgecolor='black', linewidth=.1)
        hist_shift.title.set_text('\nHistogram of Shifted Image, PSNR=' + str(psnr(self.cv_img, self.cv_imgShift)))

        self.figure_histSection.tight_layout()
        self.canvas_histSection.draw()

    def getImg(self):
        img_path, type = QFileDialog.getOpenFileName(self, 'Open Image', 'C:\\Users\\Antik\\Desktop\\code\\riwProject\\images', 'Image Files (*.jpg *.jpeg *.png)')

        if len(img_path) == 0:
            return
        
        self.flag_loadedImage = True
        self.flag_loadedWatermark = False

        self.cv_img = cv2.imread(img_path, 0)
        self.cv_img = cv2.resize(self.cv_img,(self.width_img, self.height_img))

        self.showImage(self.cv_img, self.width_img/3, self.height_img/3, self.label_imgOrig)
        self.setHistogramsSection()

    def initUI(self):
        ### IMAGE SECTION ###
        grid_image = QVBoxLayout()
        verticalSpacer = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Expanding)

        label_grid_image_1 = QLabel('Original Image')
        label_grid_image_1.setAlignment(Qt.AlignCenter)
        self.label_imgOrig = QLabel('None')
        self.label_imgOrig.setAlignment(Qt.AlignCenter)

        label_grid_image_2 = QLabel('Histogram Shifted Image')
        label_grid_image_2.setAlignment(Qt.AlignCenter)
        self.label_imgShift = QLabel('None')
        self.label_imgShift.setAlignment(Qt.AlignCenter)

        label_grid_image_3 = QLabel('Difference')
        label_grid_image_3.setAlignment(Qt.AlignCenter)
        self.label_imgDiff = QLabel('None')
        self.label_imgDiff.setAlignment(Qt.AlignCenter)

        grid_image.addWidget(label_grid_image_1)
        grid_image.addWidget(self.label_imgOrig)
        grid_image.addItem(verticalSpacer)  

        grid_image.addWidget(label_grid_image_2)
        grid_image.addWidget(self.label_imgShift)
        grid_image.addItem(verticalSpacer)  

        grid_image.addWidget(label_grid_image_3)
        grid_image.addWidget(self.label_imgDiff)
        grid_image.addItem(verticalSpacer)  

        ### IMAGE HISTOGRAM SECTION ###
        grid_imgHist = QVBoxLayout()

        grid_imgHist.addWidget(self.canvas_histSection)
        grid_imgHist.addWidget(self.toolbar_histSection)
    
        ### WATERMARK SECTION ###
        grid_watermark = QGridLayout()

        label_grid_watermark_1 = QLabel('Selected Watermark')
        label_grid_watermark_1.setAlignment(Qt.AlignCenter)
        self.label_watermarkEmb = QLabel('None')
        self.label_watermarkEmb.setAlignment(Qt.AlignCenter)

        label_grid_watermark_2 = QLabel('Watermarked Image')
        label_grid_watermark_2.setAlignment(Qt.AlignCenter)
        self.label_imgEmbed = QLabel('None')
        self.label_imgEmbed.setAlignment(Qt.AlignCenter)

        label_grid_watermark_3 = QLabel('Extracted Watermark')
        label_grid_watermark_3.setAlignment(Qt.AlignCenter)
        self.label_watermarkExt = QLabel('None')
        self.label_watermarkExt.setAlignment(Qt.AlignCenter)

        label_grid_watermark_4 = QLabel('Restored Image')
        label_grid_watermark_4.setAlignment(Qt.AlignCenter)
        self.label_imgRestored = QLabel('None')
        self.label_imgRestored.setAlignment(Qt.AlignCenter)

        grid_watermark.addWidget(label_grid_watermark_1, 1, 1)
        grid_watermark.addWidget(self.label_watermarkEmb, 2, 1)
        grid_watermark.addItem(verticalSpacer, 3, 1)

        grid_watermark.addWidget(label_grid_watermark_2, 4, 1)
        grid_watermark.addWidget(self.label_imgEmbed, 5, 1)

        grid_watermark.addWidget(label_grid_watermark_3, 1, 2)
        grid_watermark.addWidget(self.label_watermarkExt, 2, 2)
        grid_watermark.addItem(verticalSpacer, 3, 2)

        grid_watermark.addWidget(label_grid_watermark_4, 4, 2)
        grid_watermark.addWidget(self.label_imgRestored, 5, 2)

        grid_watermark.addWidget(self.canvas_histWatermark, 6, 1, 1, 2)
        grid_watermark.addWidget(self.toolbar_histWatermark, 7, 1, 1, 2)

        ### MAIN SECTION ###
        frame = QGridLayout()
        frame.setHorizontalSpacing(2)

        button_loadImage = QPushButton('Load Image')
        button_loadWatermark = QPushButton('Load and Embed Watermark')
        button_extractWatermark = QPushButton('Extract Watermark')

        button_loadImage.clicked.connect(self.getImg)
        button_loadWatermark.clicked.connect(self.getWatermark)
        button_extractWatermark.clicked.connect(self.extractWatermark)

        frame.addWidget(button_loadImage, 0, 0, 1, 1)
        frame.addWidget(button_loadWatermark, 0, 3, 1, 1)
        frame.addWidget(button_extractWatermark, 0, 4, 1, 1)

        frame.addLayout(grid_image, 1, 0, 10, 1)
        frame.addLayout(grid_imgHist, 1, 1, 10, 2)
        frame.addLayout(grid_watermark, 1, 3, 10, 2)

        ### FINALIZING ###
        self.setLayout(frame)
        
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)

        self.showMaximized()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont('Open Sans')
    font.setPixelSize(14)
    app.setFont(font)
    ex = App()
    sys.exit(app.exec_())