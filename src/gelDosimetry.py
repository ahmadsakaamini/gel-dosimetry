from pydicom import dcmread
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from scipy.signal import wiener
from scipy.signal import medfilt2d
from scipy.optimize import least_squares
from scipy.ndimage.interpolation import shift
from scipy.ndimage import generic_filter
from scipy import ndimage
from tabulate import tabulate
import pymedphys

class gelDosimetry:
    
    def __init__(self):
        self.pre_ct_folder_path = 'CT/preCT/'
        self.post_ct_folder_path = 'CT/postCT/'
        self.calculated_folder_path = 'Dose/'
        self.dcm_file = '*.dcm'
        
        self.calculated_dicom_data, self.calculated_dicom_pixel_arrays = None, None
        self.scaled_calculated_dicom_pixel_arrays = None
        
        self.preCT_dicom_data_list, self.preCT_dicom_arrays_list, self.preCT_dicom_arrays, self.preCT_dicom_arrays_list_hu, self.preCT_dicom_arrays_hu = None, None, None, None, None
        self.preCT_dicom_arrays_hu_registered = None
        self.trueCT_dicom_arrays_hu = None
        
        self.postCT_dicom_data_list, self.postCT_dicom_arrays_list, self.postCT_dicom_arrays, self.postCT_dicom_arrays_list_hu, self.postCT_dicom_arrays_hu = None, None, None, None, None
        
        self.bg_slice_n = 0
        self.irr_slice_n = 0
        self.calc_slice_n = 0
        
        self.iph_s_1, self.iph_e_1, self.ipw_s_1, self.ipw_e_1 = None, None, None, None
        self.iph_s_2, self.iph_e_2, self.ipw_s_2, self.ipw_e_2 = None, None, None, None
        
        self.flip_arrays = False
        
        self.hu_bg_slice = None
        self.hu_irr_slice = None
        self.hu_true_slice = None
        self.hu_true_slice_masked = None
        self.calc_slice = None
        self.scaled_calc_slice = None
        
        self.ofs_1, self.ofs_2 = None, None
        self.scaled_calc_slice_small = None
        self.hu_true_slice_small = None
        self.filtered_hu_true_slice_small = None
        
        self.binned_hu_scaled_calc, self.hu_bins, self.scaled_calc_medians = None, None, None
        self.hu, self.scaled_calc_median = None, None
        
        self.fitting_function = None
        self.fit_params = None
        
        self.hu_array = None
        self.hu_calibrated_array = None
        self.hu_calibrated_array_filtered = None
        self.calc_array = None

        
    def dicom_processor(self, folder_path):
        dicom_data = []
        dicom_pixel_arrays = {}
        for i, dcm in enumerate(sorted(glob.glob(folder_path + self.dcm_file))):
            ds = dcmread(dcm)
            dicom_data.append(ds)
            if ds.Modality == 'CT':
                dicom_pixel_arrays[i] = {}
                dicom_pixel_arrays[i] = ds.pixel_array
                dicom_pixel_arrays[i] = dicom_pixel_arrays[i].astype(np.float64)
            elif ds.Modality == 'RTDOSE':
                for j, arr in enumerate(ds.pixel_array):
                    dicom_pixel_arrays[j] = {}
                    dicom_pixel_arrays[j] = arr
                    dicom_pixel_arrays[j] = dicom_pixel_arrays[j].astype(np.float64)
            else:
                for j, arr in enumerate(ds.pixel_array):
                    dicom_pixel_arrays[j] = {}
                    dicom_pixel_arrays[j] = arr
                    dicom_pixel_arrays[j] = dicom_pixel_arrays[j].astype(np.float64)
        return dicom_data, dicom_pixel_arrays
    
    
    def dicom_dir_processor(self, folder_path):
        dicom_data = []
        dicom_pixel_arrays = []
        for i, dcm in enumerate(sorted(glob.glob(folder_path + self.dcm_file))):
            ds = dcmread(dcm)
            dicom_data.append(ds)
            if ds.Modality == 'CT':
                dicom_pixel_arrays.append(ds.pixel_array.astype(np.float64))
            elif ds.Modality == 'RTDOSE':
                for j, arr in enumerate(ds.pixel_array):
                    np.append(dicom_pixel_arrays, arr.astype(np.float64))
        return dicom_data, dicom_pixel_arrays
    

    def ct_image_average(self, directory):
        directories = sorted([x[0] for x in os.walk(directory)])[1:]
        dicom_data_list = []
        dicom_arrays_list = []
        for i, subdir in enumerate(directories):
            dicom_data, dicom_arrays = self.dicom_dir_processor(subdir+'/')
            dicom_data_list.append(dicom_data)
            dicom_arrays_list.append(dicom_arrays)
        dicom_data_list = np.array(dicom_data_list)
        dicom_arrays_list = np.array(dicom_arrays_list)
        dicom_arrays_averaged = np.average(dicom_arrays_list, axis=0)

        dicom_arrays_list_hu = []
        for i, scan in enumerate(dicom_data_list):
            dicom_arrays_list_hu.append(self.get_pixels_hu(scan))
        dicom_arrays_list_hu = np.array(dicom_arrays_list_hu)
        dicom_arrays_averaged_hu = np.average(dicom_arrays_list_hu, axis=0)
        return dicom_data_list, dicom_arrays_list, dicom_arrays_averaged, dicom_arrays_list_hu, dicom_arrays_averaged_hu
    
    
    def image_resize_offsets(self, dicom_pixel_array, image_length_ofs_1, image_length_ofs_2, image_width_ofs_1, image_width_ofs_2):
        image_length, image_width = dicom_pixel_array[0].shape[0], dicom_pixel_array[0].shape[1]
        image_length_half, image_width_half = int(image_length/2), int(image_width/2)
        s_y, e_y, s_x, e_x = image_length_half + image_length_ofs_1, image_length_half + image_length_ofs_2, image_width_half + image_width_ofs_1, image_width_half + image_width_ofs_2
        return s_y, e_y, s_x, e_x
    
    
    def auto_image_resize_offsets(self, calculated_pixel_arrays, measured_pixel_arrays, slice='', reg_offset_h=0, reg_offset_w=0):
        image_pixel_offset = int(round(calculated_pixel_arrays[0].shape[1] / 2))
        image_pixel_height = int((measured_pixel_arrays[0].shape[1]) / 2) 
        iph_s, iph_e = (image_pixel_height - image_pixel_offset) + reg_offset_h, (image_pixel_height + image_pixel_offset) + reg_offset_h
        image_pixel_width = int((measured_pixel_arrays[0].shape[0]) / 2) 
        ipw_s, ipw_e = (image_pixel_width - image_pixel_offset) + reg_offset_w, (image_pixel_width + image_pixel_offset) + reg_offset_w
        if not slice:
            slice = int(round(len(measured_pixel_arrays)/2))
        return slice, iph_s, iph_e, ipw_s, ipw_e
    
    
    def registration_offsets(self, dicom_data, z_shift, y_shift, x_shift, roll_angle):
        if 'SliceThickness' in dicom_data[0]:
            slice_thickness = float(dicom_data[0].SliceThickness)
        if 'SliceThickness' in dicom_data[0][0]:
            slice_thickness = float(dicom_data[0][0].SliceThickness)
        z_slice_shift = round(((z_shift/10)/slice_thickness), 3)
        return z_slice_shift, y_shift, x_shift, roll_angle
        
    
    def image_shift(self, array_3d, z_slice_shift=0, y_shift=0, x_shift=0):
        if (abs(z_slice_shift) > 0) or (abs(y_shift) > 0) or (abs(x_shift) > 0):
            array_shifted = shift(array_3d, (z_slice_shift, y_shift, x_shift), cval=0, mode='grid-constant')
            return array_shifted
        else:
            return array_3d

    
    def image_roll(self, array_3d, roll_angle=0):
        if abs(roll_angle) > 0:
            array_rolled = np.empty(array_3d.shape)
            for i, slice in enumerate(array_3d):
                slice = ndimage.rotate(slice, -0.1, reshape=False, cval=0, mode='grid-constant')
                np.append(array_rolled, slice)
            return array_rolled
        else:
            return array_3d
        

    def pixel_varaiations(self, p_array_list, slice, ofs1=None, ofs2=None, ofs3=None, ofs4=None, row=None, pixel=None):
        p_array = []
        for i, x in enumerate(p_array_list):
            p_array.append(p_array_list[i][slice][ofs1:ofs2, ofs3:ofs4][row][pixel])
        return p_array


    def pixel_variations_plot_stats(self, calculated_pixel_arrays, dicom_arrays_list_hu, stats=True, slice=None, ofs1=None, ofs2=None, ofs3=None, ofs4=None):
        pixel_location_offsets = [0, -20, 20, 0, 0]
        pixels_list = []
        if not slice:
            slice = int(round(len(dicom_arrays_list_hu[0]) / 2))
        image_pixel_halfsize = int(round(calculated_pixel_arrays[0].shape[1] / 2))

        number_of_scans = len(dicom_arrays_list_hu) + 1
        x = np.arange(1, number_of_scans, 1)

        for i, ofs in enumerate(pixel_location_offsets):
            row_ofs, pixel_ofs = pixel_location_offsets[i], pixel_location_offsets[i-2]
            pixels_from_scans = self.pixel_varaiations(
                dicom_arrays_list_hu, slice=slice, ofs1=ofs1, ofs2=ofs2, ofs3=ofs3, ofs4=ofs4, row=image_pixel_halfsize + row_ofs, pixel=image_pixel_halfsize + pixel_ofs)
            if stats:
                self.data_plot_stats(pixels_from_scans, plot=False)
            pixels_list.append(pixels_from_scans)

        plt.figure(figsize=(10, 6))
        plt.title('Pre CT')
        plt.xlabel("Scan Number")
        plt.ylabel("HU")
        plt.plot(x, pixels_list[0], 'o-', label='middle_pixel')
        plt.plot(x, pixels_list[1], 'o-', label='top_middle_pixel')
        plt.plot(x, pixels_list[2], 'o-', label='bottom_middle_pixel')
        plt.plot(x, pixels_list[3], 'o-', label='left_middle_pixel')
        plt.plot(x, pixels_list[4], 'o-', label='right_middle_pixel')
        plt.legend()
        plt.grid()
        plt.show()

    
    def get_pixels_hu(self, scans):
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        # image[image < 0] = 0

        return np.array(image, dtype=np.int16)
    
    
    def dose_pixel_gy(self, dicom_data):
        dose_grid_scaling_factor = float(dicom_data[0].DoseGridScaling)
        scaled_dicom_pixel_arrays = {}
        for i, x in enumerate(dicom_data[0].pixel_array):
            scaled_dicom_pixel_arrays[i] = np.copy(dicom_data[0].pixel_array[i]) 
            scaled_dicom_pixel_arrays[i] = scaled_dicom_pixel_arrays[i] * dose_grid_scaling_factor
        return scaled_dicom_pixel_arrays

    
    def dicom_slicer(self, dicom_pixel_arrays, slice_number, s_y, e_y, s_x, e_x):
        slice = dicom_pixel_arrays[slice_number][s_y:e_y, s_x:e_x]
        return slice
    
    
    def find_slice(self, pixel_array, o1=None, o2=None, o3=None, o4=None):
        max_dose = []
        max_dose_slice = []
        range_1 = int(round(len(pixel_array) / 2 - 10))
        range_2 = int(round(len(pixel_array) / 2 + 10))
        for i, slice in enumerate(pixel_array):
            if i > range_1 and i < range_2:
                slice_cal = pixel_array[i][o1:o2, o3:o4]
                max_dose.append(np.mean(slice_cal))
                max_dose_slice.append(i)

        dose_slice_n = max_dose_slice[np.argmax(max_dose, axis=0)]
        # print('dose_slice_n: ', dose_slice_n)
        return dose_slice_n
    
    
    def data_plot_stats(self, pixel_array, plot=True, plt_title='Title', plt_clim=(0,25), plt_cmap='hot', extent=[0, 0, 0, 0], extent_state=False, stats=True):
        
        if plot:
            plt.figure()
            plt.title(plt_title)
            if extent_state:
                plt.imshow(pixel_array, cmap=plt_cmap, clim=plt_clim, extent=extent, interpolation='nearest')
            elif extent_state == False:
                plt.imshow(pixel_array, cmap=plt_cmap, clim=plt_clim, interpolation='nearest')
            plt.colorbar()
            plt.show()

        if stats:
            print("min: ", np.min(pixel_array))
            print("max: ", np.max(pixel_array))
            print("median: ", np.median(pixel_array))
            print("mean: ", np.mean(pixel_array))
            print("std: ", np.std(pixel_array))

        print('----------')
      
        
    def multi_plot(self, data, data_labels, rows, columns, cmap='hot', clim=(0, 25), figsioze=(15, 15), extent=[0, 0, 0, 0], extent_state=False):
        fig = plt.figure(figsize=figsioze)
        for i, x in enumerate(data):
            fig.add_subplot(rows, columns, i+1)
            if extent_state:
                plt.imshow(data[i], cmap=cmap, clim=clim, extent=extent, interpolation='nearest')
            elif extent_state == False:
                plt.imshow(data[i], cmap=cmap, clim=clim, interpolation='nearest')
            plt.imshow(data[i], cmap=cmap, clim=clim, interpolation='nearest')
            plt.colorbar()
            plt.title(data_labels[i])
        plt.show()
        
    
    def flip(self, item_1, item_2):
        x = item_2 if self.flip_arrays else item_1
        y = item_1 if self.flip_arrays else item_2
        return x, y
        
            
    def data_stats(self, data, data_labels):
        stat_headers = ["Data", "Min", "Max", "Mean", "Median", "Std"]
        stats = []
        for i, stat in enumerate(data):
            stat_data = []
            stat_data.append(data_labels[i])
            stat_data.append(np.nanmin(stat))
            stat_data.append(np.nanmax(stat))
            stat_data.append(np.mean(stat))
            stat_data.append(np.median(stat))
            stat_data.append(np.std(stat))
            stats.append(stat_data)

        print(tabulate(stats, headers=stat_headers, tablefmt="orgtbl", stralign='right'))
        
        
    def create_circular_mask(self, h, w, center=None, radius=None):
        
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
            print('center:', center)
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])
            print('radius:', radius)

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
    
            
    def zero_offset_pixel_array(self, pixel_array):
        pixel_array = pixel_array + (-1*np.min(pixel_array))
        return pixel_array
    
    
    def normalize_pixel_array(self, master_array, norm_array):
        norm_max = np.max(norm_array)
        master_max = np.max(master_array)
        norm_factor = master_max / norm_max
        norm_array = norm_array * norm_factor
        return norm_array
    

    def image_filtering_offset(self, hu_slice_array, hu_min=0, hu_max=30):
        pixel_range = int(round(len(hu_slice_array) / 2))
        for i in range(pixel_range):
            if i > 0 and i < pixel_range:
                if (np.min(hu_slice_array[i:-i, i:-i]) >= hu_min) and (np.max(hu_slice_array[i:-i, i:-i]) < hu_max):
                    # print('ofs index, min, max: ', i, ',', np.min(hu_slice_array[i:-i, i:-i]), ',', np.max(hu_slice_array[i:-i, i:-i]))
                    ofs_1, ofs_2 = i, -i
                    return ofs_1, ofs_2
    
    
    def mean_filter(self, image, kernel_size, plot=True, filter_type=1):
        if filter_type ==1:
            filtered_image = wiener(image, (kernel_size, kernel_size))  # Filter the image
        elif filter_type == 2:
            filtered_image = generic_filter(image, function=np.nanmean, size=kernel_size)
        if plot:
            f, (plot1, plot2) = plt.subplots(1, 2)
            plot1.imshow(image, cmap='hot')
            plot2.imshow(filtered_image, cmap='hot')
            plt.show()
        return filtered_image
    
    def nanmean_filter(input_array, *args, **kwargs):
        """
        Arguments:
        ----------
        input_array : ndarray
            Input array to filter.
        size : scalar or tuple, optional
            See footprint, below
        footprint : array, optional
            Either `size` or `footprint` must be defined.  `size` gives
            the shape that is taken from the input array, at every element
            position, to define the input to the filter function.
            `footprint` is a boolean array that specifies (implicitly) a
            shape, but also which of the elements within this shape will get
            passed to the filter function.  Thus ``size=(n,m)`` is equivalent
            to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
            of dimensions of the input array, so that, if the input array is
            shape (10,10,10), and `size` is 2, then the actual size used is
            (2,2,2).
        output : array, optional
            The `output` parameter passes an array in which to store the
            filter output. Output array should have different name as compared
            to input array to avoid aliasing errors.
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The `mode` parameter determines how the array borders are
            handled, where `cval` is the value when mode is equal to
            'constant'. Default is 'reflect'
        cval : scalar, optional
            Value to fill past edges of input if `mode` is 'constant'. Default
            is 0.0
        origin : scalar, optional
            The `origin` parameter controls the placement of the filter.
            Default 0.0.

        See also:
        ---------
        scipy.ndimage.generic_filter
        """
        return generic_filter(input_array, function=np.nanmean, *args, **kwargs)
    
    
    def median_filter(self, image, kernel_size, plot=True):
        filtered_image = medfilt2d(image, kernel_size=kernel_size) 
        if plot:
            f, (plot1, plot2) = plt.subplots(1, 2)
            plot1.imshow(image, cmap='hot')
            plot2.imshow(filtered_image, cmap='hot')
            plt.show()
        return filtered_image


    # binning measured with calc
    def binning(self, key_array, val_array, bin_decimals = 1, plot=False):
        binned_key_vals = {}
        counter = 0
        counter2 = 0
        for i, x_array in enumerate(key_array):
            for j, y_pixel in enumerate(x_array):
                y_pixel = round(y_pixel, bin_decimals)
                if y_pixel in binned_key_vals:
                    binned_key_vals[y_pixel].append(val_array[i][j])
                else:
                    binned_key_vals[y_pixel] = []
                    binned_key_vals[y_pixel].append(val_array[i][j])
                
        #         # Counters    
        #         if binned_key_vals[y_pixel]:
        #             counter += 1
        #         else:
        #             counter2 += 1
        
        # print('counters: ', counter, counter2)
                    
        if plot:
            self.plot_scatter(key_array, val_array, plot_title='Calculated Dose - Binned vs HU', plot_xlabel='HU', plot_ylabel='Dose', plot_fig_size=(10, 6), marker='ro')
        return binned_key_vals, key_array, val_array


    def medians(self, binned_dict, plot=False):
        array_1 = []
        array_2 = []

        for i, x in enumerate(binned_dict):
            array_1.append(x)
            array_2.append(np.median(binned_dict[x]))
        array_1 = np.array(array_1)
        array_2 = np.array(array_2)

        if plot:
            self.plot_scatter(array_1, array_2, plot_title='Median Calculated Dose vs HU', plot_xlabel='HU', plot_ylabel='Dose', plot_fig_size=(10, 6), marker='ro')
        return array_1, array_2
    
    
    def plot_scatter(self, x_array, y_array, plot_title='No Title', plot_xlabel='X', plot_ylabel='Y', plot_fig_size=(10, 6), marker='ro'):
        plt.figure(figsize=plot_fig_size)
        plt.title(plot_title)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.plot(x_array, y_array, marker)
        plt.grid()
        plt.show()
    
    
    def tan_h(self, params, x):
        return params[0] + params[1] * np.tanh((params[2] * x) - params[3])
    
    
    def arc_tan(self, params, x):
        return params[0] + params[1] * np.arctan((params[2] * x) - params[3])
    
    
    def sigmoid(self, params, x):
        return params[0] / (1 + np.exp(- params[1] * (x - params[2])))
    
    
    def gen_sigmoid(self, params, x):
        # a, b, gamma, phi, k, v
        return params[0] + ((params[1] - params[0]) / ((1 + params[2] * np.exp(-params[3] * (x - params[4])))**(1/params[5])))
    
    
    def arc_tan_h(self, params, x):
        return (np.arctanh((x - params[0]) / params[1]) + params[3]) / (params[2])
    
    
    def tan(self, params, x):
        return (np.tan((x - params[0]) / params[1]) + params[3]) / (params[2])
    
    
    def a_sigmoid(self, params, x):
        return params[2] - ((np.log( (params[0] / x) - 1)) / params[1])
    
    
    def a_gen_sigmoid(self, params, x):
        return params[4] - ((np.log((((params[1] - params[0]) / (x - params[0]))**5 - 1) / (params[2]))) / params[3])
            
        
    def fit_functions(self, fitting_function, params, x):
        if fitting_function == 1:
            return self.tan_h(params, x)
        elif fitting_function == 2:
            return self.arc_tan(params, x)
        elif fitting_function == 3:
            return self.sigmoid(params, x)
        elif fitting_function == 4:
            return self.gen_sigmoid(params, x)
        
        
    def a_fit_functions(self, fitting_function, params, x):
        if fitting_function == 1:
            return self.arc_tan_h(params, x)
        elif fitting_function == 2:
            return self.tan(params, x)
        elif fitting_function == 3:
            return self.a_sigmoid(params, x)
        elif fitting_function == 4:
            return self.a_gen_sigmoid(params, x)
            
        
    def optimize(self, params, fitting_function, xs, ys):
        if fitting_function == 1:
            return self.tan_h(params, xs) - ys
        elif fitting_function == 2:
            return self.arc_tan(params, xs) - ys
        elif fitting_function == 3:
            return self.sigmoid(params, xs) - ys
        elif fitting_function == 4:
            return self.gen_sigmoid(params, xs) - ys
        
        
    def a_optimize(self, params, fitting_function, xs, ys):
        if fitting_function == 1:
            return self.arc_tan_h(params, ys) - xs
        elif fitting_function == 2:
            return self.tan(params, ys) - xs
        elif fitting_function == 3:
            return self.a_sigmoid(params, ys) - xs
        elif fitting_function == 4:
            return self.a_gen_sigmoid(params, ys) - xs
               
    
    def residuals(self, params, fitting_function, xs, ys):
        return least_squares(self.optimize, params, args=(fitting_function, xs, ys), verbose=0, bounds=(0,100))
        # return least_squares(self.a_optimize, params, args=(fitting_function, xs, ys))
    
    
    def plot_calibration_curve(self, x_array, y_array, params_0, fitting_function):
        residuals = self.residuals(params_0, fitting_function, x_array, y_array)
        params_fit = []
        for i, param in enumerate(residuals.x):
            params_fit.append(param)
            
        if fitting_function == 1:
            print('fit function: tanh')
        elif fitting_function == 2:
            print('fit function: arctan')
        elif fitting_function == 3:
            print('fit function: sigmoid')
        elif fitting_function == 4:
            print('fit function: generalized sigmoid')
            
        mse = np.mean(residuals.fun ** 2)
            
        plt.figure(figsize=(10, 6))
        plt.title(f'Calibration Curve with MSE: {round(mse, 4)}')
        xlabel, ylabel = self.flip('HU', 'Dose')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x_array, y_array, label='{} vs {}'.format(ylabel, xlabel), marker='.', color='r', linestyle='', markersize=8)
        plt.plot(x_array, self.fit_functions(fitting_function, params_fit, x_array), marker='.', color='b', linestyle='', markersize=8, label='{} vs {} Calibration Curve'.format(ylabel, xlabel))
        # plt.plot(x_array, self.a_fit_functions(fitting_function, params_fit, x_array), marker='.', color='b', linestyle='', markersize=8, label='{} vs {} Calibration Curve'.format(ylabel, xlabel))
        plt.legend()
        plt.grid()
        plt.show()
        print('fit parameters: ', params_fit)
        print('mse: ', mse)
        print('----------')
        return params_fit
    
    
    def subplots_calibration(self, x1, y1, x2, y2, suptitle, title1, title2, x1label, y1label, x2label, y2label):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        fig.suptitle(suptitle)
        ax1.set_title(title1)
        ax2.set_title(title2)
        ax1.set(xlabel=x1label, ylabel=y1label)
        ax2.set(xlabel=x2label, ylabel=y2label)
        ax1.plot(x1, y1, marker='.', color='r', linestyle='', markersize=8)
        ax2.plot(x2, y2, marker='.', color='r', linestyle='', markersize=8)
        ax1.grid()
        ax2.grid()
        
    
    def hu_calibration(self, fitting_function, params, hu_array):
        if fitting_function == 1:
            print(fitting_function)
            hu_calibrated_array = (np.arctanh((hu_array - params[0]) / params[1]) + params[3]) / (params[2])
        elif fitting_function == 2:
            print(fitting_function)
            hu_calibrated_array = (np.tan((hu_array - params[0]) / params[1]) + params[3]) / (params[2])
            # hu_calibrated_array = self.fit_functions(fitting_function, params, hu_array)
        elif fitting_function == 3:
            print(fitting_function)
            hu_calibrated_array = ''
        elif fitting_function == 4:
            print(fitting_function)
            # hu_calibrated_array = params[4] - ( (np.log( ( ((params[1] - params[0]) / (hu_array - params[0]))**5 - 1) / (params[2]) )) / params[3] )
            hu_calibrated_array = self.fit_functions(fitting_function, params, hu_array)
        
        return hu_calibrated_array
    
    
    def hu_calibration_correction(self, hu_calibrated_array, calculated_dicom_data):
        max_count = 0
        min_count = 0
        max_pixels = {}
        min_pixels = {}
        DVHMaximumDose = float(calculated_dicom_data[0].DVHSequence[1].DVHMaximumDose)
        for i, x in enumerate(hu_calibrated_array):
            for j, y in enumerate(x):
                if y > DVHMaximumDose:
                    hu_calibrated_array[i][j] = DVHMaximumDose
                    max_count += 1
                    if i in max_pixels:
                        max_pixels[i].append(j)
                    else:
                        max_pixels[i] = []
                        max_pixels[i].append(j)
                elif y < 0:
                    hu_calibrated_array[i][j] = 0
                    min_count += 1
                    if i in min_pixels:
                        min_pixels[i].append(j)
                    else:
                        min_pixels[i] = []
                        min_pixels[i].append(j)
                    
        print('max pixels corrected: ', max_count)
        print('min pixels corrected: ', min_count)
        print('max pixels locations: ', max_pixels)
        print('min pixels locations: ', min_pixels)
        return hu_calibrated_array
    
    
    def gamma_plot_params(self, dose_ref):
        grid = 1
        xmin = -1 * dose_ref.shape[0] / 2
        xmax = dose_ref.shape[0] / 2
        ymin = -1 * dose_ref.shape[0] / 2
        ymax = dose_ref.shape[0] / 2
        extent = [xmin-grid/2, xmax+grid/2, ymin-grid/2, ymax+grid/2]
        x = np.arange(xmin, xmax, grid)
        y = np.arange(ymin, ymax, grid)
        coords = (y, x)
        return extent, coords
    
    
    def gamma_analysis(self, dose_ref, dose_eval, coords, extent, dose_percent_threshold=3, distance_mm_threshold=2, lower_percent_dose_cutoff=10, local_gamma=False, quiet=True):
        # AAPM rec 3% 2 mm but no one follows it - usually it's 3%3mm with 10% threshold
        gamma_options = {
            'dose_percent_threshold': dose_percent_threshold,
            'distance_mm_threshold': distance_mm_threshold,
            'lower_percent_dose_cutoff': lower_percent_dose_cutoff,
            'interp_fraction': 10,  # Should be 10 or more for more accurate results
            'max_gamma': 2,
            'random_subset': None,
            'local_gamma': local_gamma,
            'ram_available': 2**29,  # 1/2 GB
            'quiet': quiet
        }

        gamma_no_noise = pymedphys.gamma(
            coords, dose_ref,
            coords, dose_eval,
            **gamma_options)

        plt.figure(figsize=(7, 7))
        plt.title('Gamma Distribution')

        plt.imshow(gamma_no_noise, clim=(0, 2), extent=extent, cmap='coolwarm')
        plt.colorbar()

        plt.show()
        valid_gamma_no_noise = gamma_no_noise[~np.isnan(gamma_no_noise)]
        no_noise_passing = 100 * np.round(np.sum(valid_gamma_no_noise <= 1) / len(valid_gamma_no_noise), 4)

        plt.figure(figsize=(10, 6))
        plt.title(f'Gamma Histogram | Passing rate = {round(no_noise_passing, 2)}%')
        plt.xlabel('Gamma')
        plt.ylabel('Number of pixels')
        plt.hist(valid_gamma_no_noise, 20)
    
    
class gelDosimetryAnalysis(gelDosimetry):
    def load(self):
        self.calculated_dicom_data, self.calculated_dicom_pixel_arrays = self.dicom_processor(self.calculated_folder_path)
        self.preCT_dicom_data_list, self.preCT_dicom_arrays_list, self.preCT_dicom_arrays, self.preCT_dicom_arrays_list_hu, self.preCT_dicom_arrays_hu = self.ct_image_average(self.pre_ct_folder_path)
        self.postCT_dicom_data_list, self.postCT_dicom_arrays_list, self.postCT_dicom_arrays, self.postCT_dicom_arrays_list_hu, self.postCT_dicom_arrays_hu = self.ct_image_average(self.post_ct_folder_path)
        
    def scale_dose(self):
        self.scaled_calculated_dicom_pixel_arrays = self.dose_pixel_gy(self.calculated_dicom_data)
        self.calc_slice_n = self.find_slice(self.scaled_calculated_dicom_pixel_arrays, o1=None, o2=58, o3=None, o4=None)
        self.calc_slice = self.calculated_dicom_pixel_arrays[self.calc_slice_n][:58,:]
        self.scaled_calc_slice = self.scaled_calculated_dicom_pixel_arrays[self.calc_slice_n][:58, :]
        
    def register_cts(self, z_shift=0, y_shift=0, x_shift=0, roll_angle=0):
        z_slice_shift, y_shift, x_shift, roll_angle = self.registration_offsets(self.preCT_dicom_data_list, z_shift=z_shift, y_shift=y_shift, x_shift=x_shift, roll_angle=roll_angle)
        self.preCT_dicom_arrays_hu = self.image_roll(self.image_shift(self.preCT_dicom_arrays_hu, z_slice_shift=z_slice_shift, y_shift=y_shift, x_shift=x_shift), roll_angle=roll_angle)
        # self.preCT_dicom_arrays_hu_registered = self.image_roll(self.image_shift(self.preCT_dicom_arrays_hu, z_slice_shift=z_slice_shift, y_shift=y_shift, x_shift=x_shift), roll_angle=roll_angle)
        
        
    def process(self, show_ct_slices=True, show_ct_stats=True, show_pixel_variations=False, ct_dose_reg_offset_h=0, ct_dose_reg_offset_w=0):
        slice, self.iph_s_1, self.iph_e_1, self.ipw_s_1, self.ipw_e_1 = self.auto_image_resize_offsets(self.calculated_dicom_pixel_arrays, self.preCT_dicom_arrays_hu, reg_offset_h=ct_dose_reg_offset_h, reg_offset_w=ct_dose_reg_offset_w)
        slice, self.iph_s_2, self.iph_e_2, self.ipw_s_2, self.ipw_e_2 = self.auto_image_resize_offsets(self.calculated_dicom_pixel_arrays, self.postCT_dicom_arrays_hu, reg_offset_h=0, reg_offset_w=0)
        
        data = [self.preCT_dicom_arrays_hu[slice][self.iph_s_1:self.iph_e_1, self.ipw_s_1:self.ipw_e_1], 
                self.postCT_dicom_arrays_hu[slice][self.iph_s_1:self.iph_e_1, self.ipw_s_1:self.ipw_e_1],
                self.scaled_calc_slice]
        data_labels = ['Pre CT', 'Post CT', 'Scaled Calc Dose Slice']
        if show_ct_slices:
            self.multi_plot(data, data_labels, rows=1, columns=3, cmap='hot', clim=(0, 25), figsioze=(18, 6))  
        if show_ct_stats:
            self.data_stats(data, data_labels)
        if show_pixel_variations:
            self.pixel_variations_plot_stats(self.calculated_dicom_pixel_arrays, self.preCT_dicom_arrays_list_hu, stats=False, slice=51, ofs1=self.iph_s_1, ofs2=self.iph_e_1, ofs3=self.ipw_s_1, ofs4=self.ipw_e_1)
            self.pixel_variations_plot_stats(self.calculated_dicom_pixel_arrays, self.postCT_dicom_arrays_list_hu, stats=False, slice=51, ofs1=self.iph_s_1, ofs2=self.iph_e_1, ofs3=self.ipw_s_1, ofs4=self.ipw_e_1)
            
        
    def subtract(self):
        # if self.preCT_dicom_arrays_hu_registered:
        #     self.trueCT_dicom_arrays_hu = self.postCT_dicom_arrays_hu - self.preCT_dicom_arrays_hu_registered
        # else:
        #     self.trueCT_dicom_arrays_hu = self.postCT_dicom_arrays_hu - self.preCT_dicom_arrays_hu
        self.trueCT_dicom_arrays_hu = self.postCT_dicom_arrays_hu - self.preCT_dicom_arrays_hu
            
        self.irr_slice_n = self.find_slice(self.trueCT_dicom_arrays_hu, o1=self.iph_s_1, o2=self.iph_e_1, o3=self.ipw_s_1, o4=self.ipw_e_1)
        self.hu_bg_slice = self.preCT_dicom_arrays_hu[self.irr_slice_n][self.iph_s_1:self.iph_e_1, self.ipw_s_1:self.ipw_e_1]
        self.hu_irr_slice = self.postCT_dicom_arrays_hu[self.irr_slice_n][self.iph_s_1:self.iph_e_1, self.ipw_s_1:self.ipw_e_1]
        self.hu_true_slice = self.trueCT_dicom_arrays_hu[self.irr_slice_n][self.iph_s_1:self.iph_e_1, self.ipw_s_1:self.ipw_e_1]
        
        
    def apply_mask(self, radius=None):
        h, w = self.hu_true_slice.shape[:2]
        mask = self.create_circular_mask(h, w, radius=radius)
        self.hu_true_slice_masked = self.hu_true_slice.copy()
        self.hu_true_slice_masked[~mask] = 0 # np.nan # 0.00000001 # 0 will result with divide by zero when mean filter is applied
        
        
    def show_slices(self, show_plots=True, show_data_stats=True):
        data = [self.hu_bg_slice, self.hu_irr_slice, self.hu_true_slice, self.hu_true_slice_masked]
        data_labels = ['Pre CT Slice (HU)', 'Post CT Slice (HU)', 'True CT Slice (HU)', 'Masked True CT Slice (HU)']
        
        if show_plots:
            self.multi_plot(data, data_labels, rows=2, columns=2, cmap='hot', clim=(0, 25), figsioze=(12, 12)) 
        if show_data_stats:              
            self.data_stats(data, data_labels)
            
            
    def filter(self, show_filtered_images=True, show_stats=True, filter_kernel_size=3, use_masked_image=False):
        self.ofs_1, self.ofs_2 = self.image_filtering_offset(self.hu_true_slice_masked if use_masked_image else self.hu_true_slice)

        self.scaled_calc_slice_small = self.scaled_calc_slice[self.ofs_1:self.ofs_2, self.ofs_1:self.ofs_2]
        # self.hu_true_slice_small = self.hu_true_slice[self.ofs_1:self.ofs_2, self.ofs_1:self.ofs_2]
        self.hu_true_slice_small = self.hu_true_slice_masked[self.ofs_1:self.ofs_2, self.ofs_1:self.ofs_2] if use_masked_image else self.hu_true_slice[self.ofs_1:self.ofs_2, self.ofs_1:self.ofs_2]

        self.filtered_hu_true_slice_small = self.mean_filter(self.hu_true_slice_small, filter_kernel_size, plot=False)
        
        genereic_filtered_hu_true_slice_small = self.mean_filter(self.hu_true_slice_small, filter_kernel_size, plot=False, filter_type=2)

        if show_filtered_images:
            self.multi_plot([self.hu_true_slice_small, self.filtered_hu_true_slice_small, genereic_filtered_hu_true_slice_small],
                            ['True CT Slice (HU)', 'Mean Filtered True CT Slice (HU)', 'Generic Mean Filter True CT Slice (HU)'], 
                            rows=1, columns=3, cmap='hot', clim=(0, 25), figsioze=(12, 4))  
        if show_stats:
            self.data_stats([self.hu_true_slice, self.hu_true_slice_small, self.filtered_hu_true_slice_small, genereic_filtered_hu_true_slice_small],
                            ['hu_true_slice', 'hu_true_slice_small', 'filtered_hu_true_slice_small', 'genereic_filtered_hu_true_slice_small'])
            
            
    def fit(self, show_plots=True, use_filtered_data=True, fitting_function=1, binning_method=1, bin_decimals=1):
        self.fitting_function = fitting_function
        self.binning_method = binning_method
        
        # binning method uses HU bins (dose is binned)
        if self.binning_method == 1:        
            # self.hu_bins, self.scaled_calc_medians is equal to (self.filtered_hu_true_slice_small or self.hu_true_slice_small) and self.scaled_calc_slice_small
            self.binned_hu_scaled_calc, self.hu_bins, self.scaled_calc_medians = self.binning((self.filtered_hu_true_slice_small if use_filtered_data else self.hu_true_slice_small), 
                                                                                              self.scaled_calc_slice_small,
                                                                                              bin_decimals=bin_decimals, 
                                                                                              plot=False)
            
            self.hu, self.scaled_calc_median = self.medians(self.binned_hu_scaled_calc, plot=False)
        
        # binning method uses Dose bins (HU is binned)
        elif self.binning_method == 2:
        
            self.binned_hu_scaled_calc, self.scaled_calc_medians, self.hu_bins, = self.binning(self.scaled_calc_slice_small, 
                                                                                               (self.filtered_hu_true_slice_small if use_filtered_data else self.hu_true_slice_small), 
                                                                                               bin_decimals=bin_decimals,
                                                                                               plot=False)
            
            self.scaled_calc_median, self.hu = self.medians(self.binned_hu_scaled_calc, plot=False)
        
        x1, y1 = self.flip(self.hu_bins, self.scaled_calc_medians)
        x2, y2 = self.flip(self.hu, self.scaled_calc_median)
        # title1, title2 = self.flip('Binned Calculated Dose (Gy) vs CT (HU)', 'Median Calculated Dose (Gy) vs CT (HU)')
        ytitle, xtitle = self.flip('Dose (Gy)', 'CT (HU)')
        title1 = 'Binned Calculated {} vs {}'.format(ytitle, xtitle)
        title2 = 'Median Calculated {} vs {}'.format(ytitle, xtitle)
        x1label, y1label = self.flip('CT (HU)', 'Dose (Gy)')
        x2label, y2label = self.flip('CT (HU)', 'Dose (Gy)')
        
        print(y1label, ' vs ' ,x1label)

        if show_plots:
            self.subplots_calibration(x1, y1, x2, y2,
                                      suptitle='Binning',
                                      title1=title1, title2=title2,
                                      x1label=x1label, y1label=y1label, x2label=x2label, y2label=y2label)
        
        if fitting_function == 1:
            params_0=[11, 14, 0.05, 1]
        elif fitting_function == 2:
            params_0=[0, 1, 1, 0]
        elif fitting_function == 3:
            params_0=[1, 1, 1]
        elif fitting_function == 4:
            params_0=[np.min(self.hu), np.max(self.scaled_calc_median), 1, 1, 10, 1]
        
        self.fit_params = self.plot_calibration_curve(x_array=x2, y_array=y2, params_0=params_0, fitting_function=fitting_function)
            
            
    def calibrate(self, show_plots=True, show_stats=True, filter_kernel_size=3):
        s, e = self.ofs_1, self.ofs_2
        self.hu_array = self.hu_true_slice [s:e, s:e] 
        # gel.data_plot_stats(self.hu_array, plt_title='HU True Slice')

        
        if self.flip_arrays:
            self.hu_calibrated_array = self.a_fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
        else:
            self.hu_calibrated_array = self.fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
            
        # if not self.flip_arrays:
        #     self.hu_calibrated_array = self.a_fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
        # else:
        #     self.hu_calibrated_array = self.fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
        
        # self.hu_calibrated_array = self.a_fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
        
        # self.hu_calibrated_array = self.fit_functions(fitting_function=self.fitting_function, params=self.fit_params, x=self.hu_array)
            
        # gel.data_plot_stats(self.hu_calibrated_array, plt_title='Calibrated HU True Slice')

        self.hu_calibrated_array_filtered = self.mean_filter(self.hu_calibrated_array, filter_kernel_size, plot=False, filter_type=2)
        # gel.data_plot_stats(self.hu_calibrated_array_filtered, plt_title='Calibrated HU True Slice + Mean Filtered')

        self.calc_array = self.scaled_calc_slice[s:e, s:e]
        # gel.data_plot_stats(self.calc_array, plt_title='Scaled Calc Slice')

        data = [self.hu_array, self.hu_calibrated_array, self.hu_calibrated_array_filtered, self.calc_array]
        data_labels = ['True CT Slice (HU)', 'Calibrated True CT Slice (Gy)', 'Calibrated + Filtered True CT Slice (Gy)', 'Scaled Calc Dose Slice (Gy)']
        if show_plots:
            self.multi_plot(data, data_labels, rows=2, columns=2, cmap='hot', clim=(0, 25), figsioze=(12, 12))
        if show_stats:
            self.data_stats(data, data_labels)
            
        
    def gamma(self, filtered_dose_eval=False, show_dose_plots=True, show_dose_plot_stats=True, dose_percent_threshold=3, distance_mm_threshold=2, lower_percent_dose_cutoff=10, local_gamma=False, quiet=True):
        dose_ref = self.calc_array
        dose_eval = self.hu_calibrated_array_filtered if filtered_dose_eval else self.hu_calibrated_array
        
        # if dose_eval == '':
        #     dose_eval = self.hu_calibrated_array_filtered if filtered_dose_eval else self.hu_calibrated_array
        # else:
        #     dose_eval = dose_eval
        
        print(dose_ref.shape, dose_eval.shape)
            
        dose_diff = dose_eval - dose_ref
        extent, coords = self.gamma_plot_params(dose_ref)
        
        data = [dose_ref, dose_eval, dose_diff]
        data_labels = ['Reference Dose', 'Evaluation Dose', 'Dose Difference']
        if show_dose_plots:
            self.multi_plot(data, data_labels, rows=1, columns=3, cmap='hot', clim=(0, 25), extent=extent, extent_state=True, figsioze=(15, 5))
        if show_dose_plot_stats:
            self.data_stats(data, data_labels)
            
        self.gamma_analysis(dose_ref, dose_eval, coords, extent, dose_percent_threshold=dose_percent_threshold,
                            distance_mm_threshold=distance_mm_threshold, lower_percent_dose_cutoff=lower_percent_dose_cutoff, local_gamma=local_gamma, quiet=quiet)


