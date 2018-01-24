
from vector_math import *



def tangent_point_line_circles(circle_center, circle_radius, point, clockwise):
    # Input - c circle object
    #         p point object of focus tangent line
    #         clockwise whether we are turning clockwise when spiralling from the circle to the tangent line
    # Return  tangent point on the circle 0 or 1
    # http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm
    c_x, c_y = circle_center
    p_x, p_y = point
    c_r = circle_radius

    dis = (p_x - c_x)**2 + (p_y - c_y)**2 - c_r**2  # point to circle surface distance

    if dis >= 0:
        dis = sqrt(dis)

        sign = -1 if clockwise else 1
        return Vec2(
            (c_r**2 * (p_x - c_x) + sign * c_r * (p_y - c_y) * dis) / ((p_x - c_x)**2 + (p_y - c_y)**2) + c_x,
            (c_r**2 * (p_y - c_y) - sign * c_r * (p_x - c_x) * dis) / ((p_x - c_x)**2 + (p_y - c_y)**2) + c_y,
        )
    else:
        return None

def inner_tangent_points(center_0, radius_0, center_1, radius_1, clockwise):
    big_tangent_point = tangent_point_line_circles(center_0, radius_0 + radius_1, center_1, clockwise)
    if big_tangent_point is None:
        return None
    center_to_big_tangent_x = big_tangent_point - center_0
    normalized = center_to_big_tangent_x / (radius_0 + radius_1)
    return [
        center_0 + normalized * radius_0,  # point on circle_0
        center_1 - normalized * radius_1,  # point on circle_1
    ]



def main():

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg

    from collections import namedtuple



    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Tangents!")
    win.resize(800,800)

    pg.setConfigOptions(antialias=True)
    plot = win.addPlot(title="Tangents!")
    plot.showGrid(x=True,y=True)
    plot.setAspectLocked()

    turn_circle_scatter = pg.ScatterPlotItem(pxMode=False)
    plot.addItem(turn_circle_scatter)



    class DraggableNodes(pg.GraphItem):
        def __init__(self):
            self.dragPoint = None
            self.dragOffset = None
            pg.GraphItem.__init__(self)

        def setData(self, **kwds):
            self.text = kwds.pop('text', [])
            self.data = kwds
            if 'pos' in self.data:
                self.data['pos'] = np.array(self.data['pos'])
                npts = self.data['pos'].shape[0]
                self.data['data'] = np.empty(npts, dtype=[('index', int)])
                self.data['data']['index'] = np.arange(npts)
            self.updateGraph()

        def updateGraph(self):
            pg.GraphItem.setData(self, **self.data)
            if 'pos' not in self.data:
                return
            dragged_positions = self.data['pos']
            circle_0 = dragged_positions[0]
            circle_1 = dragged_positions[1]
            update(circle_0, radius_0, circle_1, radius_1)


        def mouseDragEvent(self, ev):
            if ev.button() != QtCore.Qt.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first
                # pressed:
                pos = ev.buttonDownPos()
                pts = self.scatter.pointsAt(pos)
                if len(pts) == 0:
                    ev.ignore()
                    return
                self.dragPoint = pts[0]
                ind = pts[0].data()[0]
                self.dragOffset = self.data['pos'][ind] - pos
            elif ev.isFinish():
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return

            ind = self.dragPoint.data()[0]
            self.data['pos'][ind] = ev.pos() + self.dragOffset
            self.updateGraph()
            ev.accept()


    curve = pg.PlotDataItem(symbol='o')
    plot.addItem(curve)



    def update(center_0, radius_0, center_1, radius_1):
        circles = [ (center_0, radius_0), (center_1, radius_1) ]
        tangent_point = tangent_point_line_circles(center_0, radius_0, center_1, clockwise=True)
        tangents = inner_tangent_points(center_0, radius_0, center_1, radius_1, clockwise=False)
        if tangents is None:
            return
        points = [
            center_0,
            # tangent_point,
            tangents[0],
            tangents[1],
            center_1,
        ]
        spots = []
        for center, radius in circles:
            spots.append({'pos': center, 'size': 2*radius, 'pen': {'color': 'w', 'width': 1}, })
        turn_circle_scatter.setData(spots)

        curve.setData(
            [p[0] for p in points],
            [p[1] for p in points],
        )

    update(Vec2(1,2), 3, Vec2(7,6.5), 1)
    center_0 = Vec2(1,2)
    radius_0 = 3
    center_1 = Vec2(7,6.5)
    radius_1 = 1


    dragables = DraggableNodes()
    dragables.setData(pos=[center_0, center_1], symbol='o', size=30, symbolBrush=pg.mkBrush('#444444'))
    plot.addItem(dragables)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    main()
