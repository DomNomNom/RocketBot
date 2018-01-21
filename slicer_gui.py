import threading
import numpy as np
import json
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point
import traceback
from slicer_constants import MIN_X, MAX_X, POS, SET_SPOTS, SET_REGION


spots = [{POS: [0, 0]}, {POS: [2, 1]}]
region = None
scatter_plot = None

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


def update():
    global region
    region.setZValue(10)
    min_x, max_x = region.getRegion()
    print(json.dumps({MIN_X: min_x, MAX_X: max_x}))
    sys.stdout.flush()

def set_spots(spots_json):
    global spots
    spots = spots_json
    scatter_plot.setData(spots)

def set_region(region_json):
    global region
    min_x = region_json[MIN_X]
    max_x = region_json[MAX_X]
    if not np.isfinite(min_x): min_x = min(spot[POS][0] for spot in spots())
    if not np.isfinite(max_x): max_x = max(spot[POS][0] for spot in spots())

    region.setRegion([min_x, max_x])


def read_input():
    try:
        while True:
            try:
                line = input()
            except EOFError as e:
                return
            message = json.loads(line)
            if SET_SPOTS in message:
                set_spots(message[SET_SPOTS])
            elif SET_REGION in message:
                set_region(message[SET_REGION])
            else:
                eprint('bad message: ', message)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.flush()
        os.exit(-1)


def main():
    global region
    global scatter_plot


    # window layout
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow()
    win.setWindowTitle('Slicer')
    label = pg.LabelItem(justify='right')
    win.addItem(label)
    view_box = win.addPlot(row=1, col=0)

    region = pg.LinearRegionItem()
    region.setZValue(10)
    region.sigRegionChanged.connect(update)
    # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this
    # item when doing auto-range calculations.
    view_box.addItem(region, ignoreBounds=True)

    threading.Thread(target=read_input, daemon=True).start()

    # pg.dbg()

    scatter_plot = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    set_spots(spots)
    view_box.addItem(scatter_plot)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()
