
from vector_math import *
from collections import namedtuple

# Calculates stuff relating to tangents.
# Has a main() for debugging purposes.
# Note: "clockwise" in this file assumes X=right, Y=up
# !! If you want to keep thinking in Unreal coordinate system, think of the gamefield viewed from below, not above.

def tangent_point_line_circles(circle_center, circle_radius, point, clockwise):
    # Input - c circle object
    #         p point object of focus tangent line
    #         clockwise whether we are turning clockwise when spiralling from the circle to the tangent line
    # Return  tangent point on the circle 0 or 1
    # http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm
    c_x, c_y = circle_center
    p_x, p_y = point
    c_r = circle_radius

    to_surface_dist = (p_x - c_x)**2 + (p_y - c_y)**2 - c_r**2  # point to circle surface distance

    if to_surface_dist >= 0:
        dis = sqrt(to_surface_dist)

        sign = 1 if clockwise else -1
        return Vec2(
            # TODO: Maybe there's amore nice vector expression.
            (c_r**2 * (p_x - c_x) + sign * c_r * (p_y - c_y) * dis) / ((p_x - c_x)**2 + (p_y - c_y)**2) + c_x,
            (c_r**2 * (p_y - c_y) - sign * c_r * (p_x - c_x) * dis) / ((p_x - c_x)**2 + (p_y - c_y)**2) + c_y,
        )
    else:
        # Point is inside the circle. No solutions.
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

def outer_tangent_points(center_0, radius_0, center_1, radius_1, clockwise):
    if radius_0 == radius_1:  # Special case: Outer tangents never meet.
        if all(np.isclose(center_0, center_1)):
            right = Vec2(1.0, 0.0)  # Technically, we'd have infinite solutions. Let's pick one.
        else: # common case
            right = clockwise90degrees(normalize(center_1 - center_0))

        sign = -1 if clockwise else 1
        return [
           center_0 + (sign * radius_0) * right,
           center_1 + (sign * radius_1) * right,
        ]

    if radius_0 < radius_1:  # This changes the side on which the tangents meet
        clockwise = not clockwise
    intersection = (radius_1 * center_0 - radius_0 * center_1) / (radius_1 - radius_0)
    out = [
        tangent_point_line_circles(center_0, radius_0, intersection, clockwise),  # point on circle_0
        tangent_point_line_circles(center_1, radius_1, intersection, clockwise),  # point on circle_1
    ]
    if any(vec is None  for vec in out):
        return None
    return out

TangetPath = namedtuple('TangentPath', 'pos_0 pos_1 clockwise_0 clockwise_1 turn_center_0 turn_center_1 tangent_0 tangent_1')

def get_tangent_paths(pos_0, turn_radius_0, right_0, pos_1, turn_radius_1, right_1):
    # arguments:
    #   *_0 is the specification for the start
    #   *_1 is the specification for the end
    #   pos - Position
    #   turn_radius
    #   right - The normalized vector facing to the right of the velocity.
    #           Note: x=right, y=up. Therefore right([1, 0]) == [-1, 0]
    # returns:
    #   list of TangetPaths, not sorted.

    turn_center_r_0 = pos_0 + turn_radius_0 * right_0
    turn_center_l_0 = pos_0 - turn_radius_0 * right_0
    turn_center_r_1 = pos_1 + turn_radius_1 * right_1
    turn_center_l_1 = pos_1 - turn_radius_1 * right_1

    configs = [
        # function                turn_center_0, clockwise_0,  turn_center_1,   clockwise_1
        (inner_tangent_points,  turn_center_l_0, False,        turn_center_r_1, True       ),
        (inner_tangent_points,  turn_center_r_0, True,         turn_center_l_1, False      ),
        (outer_tangent_points,  turn_center_l_0, False,        turn_center_l_1, False      ),
        (outer_tangent_points,  turn_center_r_0, True,         turn_center_r_1, True       ),
    ]
    tangent_pairs = [
        func(turn_center_0, turn_radius_0, turn_center_1, turn_radius_1, clockwise_0)
        for (func, turn_center_0, clockwise_0, turn_center_1, _) in configs
    ]

    options = []
    for config, tangents in zip(configs, tangent_pairs):
        if tangents is None:
            continue
        (_, turn_center_0, clockwise_0, turn_center_1, clockwise_1) = config
        options.append(TangetPath(
            pos_0,
            pos_1,
            clockwise_0,
            clockwise_1,
            turn_center_0,
            turn_center_1,
            tangents[0],
            tangents[1],
        ))
    return options


def get_length_of_tangent_path(path):
    radius_0 = dist(path.turn_center_0, path.pos_0)
    radius_1 = dist(path.turn_center_1, path.pos_1)
    # Note: There is intentional asymmetry in the next two lines
    arc_length_0 = radius_0 * directional_angle(path.pos_0, path.turn_center_0, path.tangent_0, path.clockwise_0)
    arc_length_1 = radius_1 * directional_angle(path.tangent_1, path.turn_center_1, path.pos_1, path.clockwise_1)
    return arc_length_0 + dist(path.tangent_0, path.tangent_1) + arc_length_1

# class TangentVisualizer(object):
#     def __init__(self, view_area):
#         self.view_area = view_area

def main():

    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg

    from collections import namedtuple



    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Tangents!")
    win.resize(800,800)

    pg.setConfigOptions(antialias=True)
    view_area = win.addPlot(title="Tangents!")
    view_area.invertX(True)  # make the coordinate system the same as the RocketLeague ground, viewed from above
    view_area.showGrid(x=True,y=True)
    view_area.setAspectLocked()
    view_area.setXRange(-3, 10)
    view_area.setYRange(-3, 10)

    turn_circle_scatter = pg.ScatterPlotItem(pxMode=False)
    view_area.addItem(turn_circle_scatter)



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
            control_points = self.data['pos']
            update(control_points)


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
                index = pts[0].data()[0]
                self.dragOffset = self.data['pos'][index] - pos
            elif ev.isFinish():
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return

            index = self.dragPoint.data()[0]

            # When dragging the centers, drag the vel with it.
            if index%2 == 0:  # car/target
                vel = self.data['pos'][index+1] - self.data['pos'][index]
                self.data['pos'][index] = ev.pos() + self.dragOffset
                self.data['pos'][index+1] = self.data['pos'][index] + vel
            else:
                self.data['pos'][index] = ev.pos() + self.dragOffset

            self.updateGraph()
            ev.accept()


    vel_curve_0 = pg.PlotDataItem()
    vel_curve_1 = pg.PlotDataItem()
    wedge_curve_0 = pg.PlotDataItem()
    wedge_curve_1 = pg.PlotDataItem()
    tangent_curve = pg.PlotDataItem(symbol='o')
    view_area.addItem(wedge_curve_0)
    view_area.addItem(wedge_curve_1)
    view_area.addItem(tangent_curve)
    view_area.addItem(vel_curve_0)
    view_area.addItem(vel_curve_1)

    def set_data_points(view_area_data_item, points):
        view_area_data_item.setData(
            [p[0] for p in points],
            [p[1] for p in points],
        )


    def set_wedge_data(wedge_curve, outer_0, center, outer_1, clockwise):
        # finish_angle = clockwise_angle_abc(outer_0, center, outer_1)
        # if not clockwise:
        #     finish_angle *= -1
        # max_angle = positive_angle(finish_angle)
        max_angle = directional_angle(outer_0, center, outer_1, clockwise)
        points = [
            outer_0,
            center,
        ]
        step_angle = tau / 31
        rotate = clockwise_matrix(step_angle if clockwise else -step_angle)
        spoke = outer_0 - center
        for i in np.arange(step_angle, max_angle, step_angle):
            spoke = rotate.dot(spoke)
            points.append(center + spoke)
            points.append(center)
        set_data_points(wedge_curve, points)
        wedge_curve.setPen(pg.mkPen(color='#FFFFFF20'))

    def update(control_points):
        center_0, vel_0, center_1, vel_1 = control_points
        set_data_points(vel_curve_0, [center_0, vel_0])
        set_data_points(vel_curve_1, [center_1, vel_1])
        vel_0 = vel_0 - center_0
        vel_1 = vel_1 - center_1

        right_0 = normalize(clockwise90degrees(vel_0))
        right_1 = normalize(clockwise90degrees(vel_1))

        radius_0 = 0.5 * mag(vel_0)
        radius_1 = 0.5 * mag(vel_1)

        turn_point_r_0 = center_0 + radius_0 * right_0
        turn_point_l_0 = center_0 - radius_0 * right_0
        turn_point_r_1 = center_1 + radius_1 * right_1
        turn_point_l_1 = center_1 - radius_1 * right_1

        turn_point_0 = turn_point_r_0
        turn_point_1 = turn_point_r_1
        tangent_paths = get_tangent_paths(center_0, radius_0, right_0, center_1, radius_1, right_1)
        if not len(tangent_paths):
            return
        tangent_paths.sort(key=get_length_of_tangent_path)
        path = tangent_paths[0]
        visualize_tangent_path(path)

    def visualize_tangent_path(path):
        points = [
            path.pos_0,
            path.turn_center_0,
            path.tangent_0,
            path.tangent_1,
            path.turn_center_1,
            path.pos_1,
        ]
        set_wedge_data(wedge_curve_0, path.pos_0, path.turn_center_0, path.tangent_0, path.clockwise_0)
        set_wedge_data(wedge_curve_1, path.tangent_1, path.turn_center_1, path.pos_1, path.clockwise_1)
        radius_0 = dist(path.turn_center_0, path.tangent_0)
        radius_1 = dist(path.turn_center_1, path.tangent_1)
        circles = [
            (path.turn_center_0, radius_0),
            (path.turn_center_1, radius_1),
        ]
        spots = []
        for center, radius in circles:
            spots.append({'pos': center, 'size': 2*radius, 'pen': {'color': 'w', 'width': 1}, 'brush': '#3333FF20',})
        turn_circle_scatter.setData(spots)

        set_data_points(tangent_curve, points)

    center_0 = Vec2(1, 2)
    center_1 = Vec2(7, 6.5)
    control_points = [
        center_0,
        center_0 + 1.5 * Vec2(0,1),

        center_1,
        center_1 + 2 * Vec2(0,1)
    ]
    update(control_points)


    dragables = DraggableNodes()
    dragables.setData(pos=control_points, symbol='o', size=30, symbolBrush=pg.mkBrush('#444444'))
    view_area.addItem(dragables)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    main()
