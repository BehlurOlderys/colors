from astropy.io import fits
import argparse
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import PySimpleGUI as sg
import glob
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

used_max = 1

available_actions = ["gui", "save_star", "save_responses"]
available_cameras = ["asi120mm"]
available_filters = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob("filters/*.csv")]

available_templates = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob("templates/*.fits")]

default_config = {
    "camera": "asi120mm",
    "filter_R": "zwo_new_R",
    "red_x": "1.0",
    "filter_G": "zwo_new_G",
    "green_x": "1.0",
    "filter_B": "zwo_new_B",
    "blue_x": "1.0",
    "star_type": "G5_+0.0_Dwarf"
}


class Range:
    def __init__(self, b=4000, e=10500, s=100): #angstroems
        self.begin_range = b
        self.end_range = e
        self.step = s
        self.xvals = np.arange(self.begin_range, self.end_range, self.step)


def normalize_with_0min(x, normal=1):
    if len(set(x)) == 1:
        return [0.5] * len(x)

    temp_x = np.asarray(x+[0])
    return normal*(x - temp_x.min()) / (np.ptp(temp_x))


def generate_camera_values(camera_file_name, range=Range()):
    file_path = os.path.join('camera_responses', camera_file_name + '.csv')
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rows_without_header = iter(csv_reader)
        next(rows_without_header)
        zipped = np.array([row[0].split(',') for row in rows_without_header]).astype(np.float)
        lam = 10*zipped[:, 0]
        val = normalize_with_0min(np.interp(range.xvals, lam, zipped[:, 1]), used_max)

    return val


def generate_filter_values(filter_file_name, range=Range()):
    file_path = os.path.join('filters', filter_file_name + '.csv')
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rows_without_header = iter(csv_reader)
        next(rows_without_header)
        zipped = np.array([row[0].split(',') for row in rows_without_header]).astype(np.float)
        lam = 10*zipped[:, 0]
        val = normalize_with_0min(np.interp(range.xvals, lam, zipped[:, 1]), used_max)

    return val


def generate_template_values(name, range=Range()):
    hdulist = fits.open(os.path.join('templates', name + '.fits'))
    lam = np.power(10, hdulist[1].data['loglam'])
    flux = hdulist[1].data['flux']
    val = normalize_with_0min(np.interp(range.xvals, lam, flux), used_max)
    return val


def get_integrated_intensity(template_values, camera_values, filter_values, range=Range(), color='violet'):
    final_curve_values = np.multiply(template_values, np.multiply(camera_values, filter_values))
    plt.plot(range.xvals, final_curve_values, color='xkcd:'+color, label='final curve')
    return integrate.cumtrapz(final_curve_values, range.xvals)[-1]


def validate_config(values):
    for k, v in values.items():
        if v is None or v is '':
            if k == '-CANVAS-':
                continue
            print(f"You did not choose {k} value!")
            exit(0)
    return values


class Application:
    def __init__(self):
        self.c = None
        self.r = None
        self.t = None
        self.g = None
        self.b = None
        self.span = Range()
        self.window = None
        self.fig = None
        self.ax = None
        self.figure_canvas_agg = None
        self.ax_legend = None
        self.used_config = default_config

    def draw_figure(self, canvas):
        if self.figure_canvas_agg is None:
            self.figure_canvas_agg = FigureCanvasTkAgg(self.fig, canvas)
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return self.figure_canvas_agg

    def get_responses(self, config):
        red_x = float(config['red_x'])
        green_x = float(config['green_x'])
        blue_x = float(config['blue_x'])

        self.c = generate_camera_values(config['camera'],     range=self.span)
        self.r = generate_filter_values(config['filter_R'],   range=self.span) * red_x
        self.g = generate_filter_values(config['filter_G'],   range=self.span) * green_x
        self.b = generate_filter_values(config['filter_B'],   range=self.span) * blue_x
        self.t = generate_template_values(config['star_type'], range=self.span)

    def get_rgb_color(self, config):
        self.get_responses(config)
        r_px = get_integrated_intensity(self.t, self.c, self.r)
        g_px = get_integrated_intensity(self.t, self.c, self.g)
        b_px = get_integrated_intensity(self.t, self.c, self.b)

        [R, G, B] = normalize_with_0min([r_px, g_px, b_px], 1)

        return R, G, B

    def plot_spectrum_values(self, val, name, color):
        self.ax.plot(self.span.xvals, val, color='xkcd:' + color, label=name)

    def plot_responses(self, config):
        if self.fig is None:
            self.fig = plt.figure(figsize=(5, 4), dpi=100)
        if self.ax is None:
            self.ax = self.fig.add_axes([0, 0, 1, 1])

        if self.ax_legend:
            self.ax.clear()
            self.ax_legend.remove()

        self.get_responses(config)
        self.plot_spectrum_values(self.c, config['camera'], 'black')
        self.plot_spectrum_values(self.r, config['filter_R'], 'red')
        self.plot_spectrum_values(self.g, config['filter_G'], 'green')
        self.plot_spectrum_values(self.b, config['filter_B'], 'blue')
        self.plot_spectrum_values(self.t, config['star_type'], 'grey')
        self.ax_legend = self.ax.legend()

    def plot_star(self, color):
        x, y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.25, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

        colors = ["black", color, "white"]
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

        if self.fig is None:
            self.fig = plt.figure(figsize=(5, 4), dpi=100)

        if self.ax is None:
            self.ax = self.fig.add_axes([0, 0, 1, 1])

        if self.ax_legend:
            self.ax_legend.remove()

        self.ax.imshow(g, interpolation='bilinear', cmap=cmap1)

    def perform_command_line(self, action):
        if action == "save_star":
            self.get_responses(self.used_config)
            rgb_color = self.get_rgb_color(self.used_config)
            self.plot_star(rgb_color)
            self.fig.savefig(self.used_config["star_output_file"])
            print(rgb_color)
        elif action == "save_responses":
            self.get_responses(self.used_config)
            self.plot_responses(self.used_config)
            self.fig.savefig(self.used_config["responses_output_file"])
        else:
            return

    def run(self, action):
        if action is not "gui":
            self.perform_command_line(action)
            return

        sg.theme('DarkAmber')
        layout = [
        [
            sg.Text('Select camera:'),
            sg.Combo(available_cameras, key='camera', default_value=self.used_config['camera']),
        ],
        [
            sg.Text('Select filter for R channel:'),
            sg.Combo(available_filters, key='filter_R', default_value=self.used_config['filter_R']),
            sg.Text('Multiplier:'),
            sg.Input(key='red_x', default_text='1.0', size=(10, )),
        ],
        [
            sg.Text('Select filter for G channel:'),
            sg.Combo(available_filters, key='filter_G', default_value=self.used_config['filter_G']),
            sg.Text('Multiplier:'),
            sg.Input(key='green_x', default_text='1.0', size=(10, )),
        ],
        [
            sg.Text('Select filter for B channel:'),
            sg.Combo(available_filters, key='filter_B', default_value=self.used_config['filter_B']),
            sg.Text('Multiplier:'),
            sg.Input(key='blue_x', default_text='1.0', size=(10, )),
        ],
        [
            sg.Text('Select star type:'),
            sg.Combo(available_templates, key='star_type', default_value=self.used_config['star_type'])
        ],
        [
            sg.Button('Check star color'), sg.Button('Plot responses')
        ],
        [
            sg.Canvas(key='-CANVAS-')
        ],
        [
            sg.Button('Exit')
        ]]

        self.window = sg.Window('Choose configuration', layout)
        while True:
            event, values = self.window.read()
            if event is None:
                break
            if event is 'Exit':
                exit(0)

            if event is 'Check star color':
                config = validate_config(values)
                print('You entered ', config)
                self.get_responses(config)
                self.plot_star(self.get_rgb_color(config))
                self.draw_figure(self.window['-CANVAS-'].TKCanvas)

            if event is 'Plot responses':
                config = validate_config(values)
                self.get_responses(config)
                self.window['-CANVAS-'].TKCanvas.delete("ALL")
                self.plot_responses(config)
                self.draw_figure(self.window['-CANVAS-'].TKCanvas)

        self.window.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Plotting star colors')
    parser.add_argument('--camera', '-c', default=default_config["camera"], choices=available_cameras)
    parser.add_argument('--filter_R', '-r', default=default_config["filter_R"], choices=available_filters)
    parser.add_argument('--filter_G', '-g', default=default_config["filter_G"], choices=available_filters)
    parser.add_argument('--filter_B', '-b', default=default_config["filter_B"], choices=available_filters)
    parser.add_argument('--red_coefficient', default=default_config["red_x"], type=float)
    parser.add_argument('--green_coefficient', default=default_config["green_x"], type=float)
    parser.add_argument('--blue_coefficient', default=default_config["blue_x"], type=float)
    parser.add_argument('--star_type', '-s', default=default_config["star_type"])
    parser.add_argument('--action', '-a', choices=available_actions, default="gui")
    parser.add_argument('--star_output_file', default="star.png")
    parser.add_argument('--list_star_types', '-l', action='store_true')
    parser.add_argument('--responses_output_file', default="responses.png")

    args = parser.parse_args()

    if args.list_star_types:
        for t in available_templates:
            print(t)
        exit(0)

    used_star_type = args.star_type.replace('\r', '').replace('\n', '').strip()
    if args.star_output_file == "star.png" and not (used_star_type == default_config["star_type"]):
        args.star_output_file = used_star_type + ".png"

    app = Application()
    app.used_config["camera"] = args.camera
    app.used_config["filter_R"] = args.filter_R
    app.used_config["filter_G"] = args.filter_G
    app.used_config["filter_B"] = args.filter_B
    app.used_config["red_x"] = args.red_coefficient
    app.used_config["green_x"] = args.green_coefficient
    app.used_config["blue_x"] = args.blue_coefficient
    app.used_config["star_type"] = used_star_type
    app.used_config["star_output_file"] = args.star_output_file
    app.used_config["responses_output_file"] = args.responses_output_file
    app.run(args.action)
