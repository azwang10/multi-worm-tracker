import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, measure, util
from joblib import Parallel, delayed
plt.rcParams['font.sans-serif'] = 'Arial'

name_from_path = lambda name: os.path.splitext(os.path.basename(name))[0]

def make_scratch_dir(input_avi, scratch_path):
	exp_name = name_from_path(input_avi)
	scratch_dir = os.path.join(scratch_path, exp_name + '-scratch')
	os.makedirs(scratch_dir, exist_ok=True)
	return scratch_dir

def convert_to_tif(input_avi, scratch_dir, frame_lim):
	tif_dir = os.path.join(scratch_dir, 'tif')
	os.makedirs(tif_dir, exist_ok=True)
	if frame_lim == None:
		os.system("ffmpeg -hide_banner -i '%s' -pix_fmt gray '%s/%%06d.tif'" % (input_avi, tif_dir))
	else:
		os.system("ffmpeg -hide_banner -i %s -pix_fmt gray -frames:v %i '%s/%%06d.tif'" % (input_avi, frame_lim, tif_dir))

def make_mask(scratch_dir):
	tif_dir = os.path.join(scratch_dir, 'tif')
	tif_list = sorted(glob.glob(os.path.join(tif_dir, '*.tif')))
	sum_arr = np.zeros(io.imread(tif_list[0]).shape, np.float64)
	indexes = np.linspace(0, len(tif_list) - 1, 100, dtype=int)
	for i in range(len(indexes)):
		sum_arr += io.imread(tif_list[indexes[i]])
	mean_arr = util.img_as_ubyte(sum_arr / len(indexes) / 255)
	io.imsave(os.path.join(scratch_dir, 'mask.tif'), mean_arr)

def show_mask(scratch_dir):
	mask = io.imread(os.path.join(scratch_dir, 'mask.tif'))
	plt.imshow(mask, cmap='Greys_r', vmin=0, vmax=255)
	return mask

def get_worms(tif, mask, sigma, thresh, area_range, ecc_range):
	tif_frame = io.imread(tif)
	tif_mean, mask_mean = np.mean(tif_frame), np.mean(mask)
	diff_frame = (mask - tif_frame / tif_mean * mask_mean).astype(np.byte)
	filtered_frame = filters.gaussian(diff_frame, sigma=sigma, preserve_range=True)
	label_img = measure.label(filtered_frame > thresh)
	props = pd.DataFrame(measure.regionprops_table(label_img, properties=['area', 'centroid', 'eccentricity']))
	worm_indexes = np.all([props['area'] > area_range[0], props['area'] < area_range[1],
						   props['eccentricity'] > ecc_range[0], props['eccentricity'] < ecc_range[1]], axis=0)
	worm_df = props.iloc[worm_indexes]
	return filtered_frame, label_img, worm_df

def check_params(scratch_dir, mask, sigma, thresh, area_range, ecc_range, rows=5):
	fig, axs = plt.subplots(rows, 4, figsize=(20, 5 * rows))
	if rows == 1:
		axs = np.expand_dims(axs, 0)
	tif_dir = os.path.join(scratch_dir, 'tif')
	tif_list = sorted(glob.glob(os.path.join(tif_dir, '*.tif')))
	indexes = np.linspace(0, len(tif_list) - 1, rows).astype(int)
	for i in range(rows):
		tif = tif_list[indexes[i]]
		diff_frame, label_img, worm_df = get_worms(tif, mask, sigma, thresh, area_range, ecc_range)
		axs[i, 0].imshow(diff_frame, vmin=0, vmax=thresh)
		axs[i, 0].set_ylabel('Frame %i' % (indexes[i] + 1))
		axs[i, 1].imshow(label_img > 0)
		axs[i, 2].imshow(np.isin(label_img, worm_df.index + 1))
		axs[i, 3].hist(worm_df['area'], label='%i blobs to %i worms' % (np.max(label_img), len(worm_df)))
		axs[i, 3].axvline(100, linestyle='--', c='k')
		axs[i, 3].set_xlim(area_range[0], area_range[1])
		axs[i, 3].legend()
		axs[i, 0].set_xticks([]); axs[i, 1].set_xticks([]); axs[i, 2].set_xticks([])
		axs[i, 0].set_yticks([]); axs[i, 1].set_yticks([]); axs[i, 2].set_yticks([])
	axs[0, 0].set_title('Mean subtracted and filtered frame')
	axs[0, 1].set_title('Thresholded frame')
	axs[0, 2].set_title('Probable worms')
	axs[0, 3].set_title('Worm sizes')
	plt.tight_layout()

def detect_worms(scratch_dir, mask, sigma, thresh, area_range, ecc_range):
	worm_dir = os.path.join(scratch_dir, 'worms')
	os.makedirs(worm_dir, exist_ok=True)

	def parallel(tif):
		num = name_from_path(tif)
		_, _, worm_df = get_worms(tif, mask, sigma, thresh, area_range, ecc_range)
		out = pd.DataFrame()
		out['frame'] = [int(num)] * len(worm_df)
		out[['x', 'y']] = worm_df[['centroid-1', 'centroid-0']].round(2).to_numpy()
		out['id'] = range(len(out))
		out.to_csv(os.path.join(worm_dir, num + '.csv'), index=False)

	tif_dir = os.path.join(scratch_dir, 'tif')
	tif_list = sorted(glob.glob(os.path.join(tif_dir, '*.tif')))
	Parallel(n_jobs=-1)(delayed(parallel)(i) for i in tif_list)

def plot_worms(scratch_dir):
	worm_dir = os.path.join(scratch_dir, 'worms')
	worm_list = sorted(glob.glob(os.path.join(worm_dir, '*.csv')))
	plt.plot([len(pd.read_csv(i)) for i in worm_list])
	plt.ylabel('Worms per frame'); plt.xlabel('Frame #')
	plt.title('Untracked worms')

def link_worms(scratch_dir, seperation):
	track_dir = os.path.join(scratch_dir, 'tracks')
	worm_dir = os.path.join(scratch_dir, 'worms')
	os.makedirs(track_dir, exist_ok=True)
	os.system('cd %s && tar -cf %s/worms.tar *.csv' % (worm_dir, scratch_dir))
	os.system('tar -xf %s/worms.tar -C %s' % (scratch_dir, track_dir))

	def link(index, track_list):
		#renumbers ids so that all ids are unique
		df1, df2 = pd.read_csv(track_list[index]), pd.read_csv(track_list[index+1])
		ids1, ids2 = sorted(set(df1['id'])), sorted(set(df2['id']))
		df1['id'] = df1['id'].replace(dict(zip(ids1, range(len(ids1)))))
		df2['id'] = df2['id'].replace(dict(zip(ids2, range(len(ids1), len(ids1) + len(ids2)))))

		#if one has no worms just ends
		if len(df1) == 0 or len(df2) == 0:
			df = pd.concat([df1, df2])
			df.to_csv(track_list[index], index=False)
			os.remove(track_list[index+1])
			return

		#get last frame of first df and first frame of second df
		sub_df1 = df1[df1['frame'] == np.max(df1['frame'])]
		arr1 = np.array(sub_df1[['x', 'y']])
		sub_df2 = df2[df2['frame'] == np.min(df2['frame'])]
		arr2 = np.array(sub_df2[['x', 'y']])

		#get a dict of ids to be replaced, every worm from the first df should only link to one worm of the second df
		diff = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]
		D = np.sqrt(np.sum(diff ** 2, axis=2))
		replace_dict = {}
		while True:
			if np.min(D) > seperation:
				break
			index2 = np.argmin(np.min(D, axis=0))
			index1 = np.argmin(D[:, index2])
			replace_dict[sub_df2['id'].iloc[index2]] = sub_df1['id'].iloc[index1]
			D[index1], D[:, index2] = np.inf, np.inf
		
		#replaces ids and overwrites files
		df2['id'] = df2['id'].replace(replace_dict)
		df = pd.concat([df1, df2])
		df.to_csv(track_list[index], index=False)
		os.remove(track_list[index+1])

	while True:
		track_list = sorted(glob.glob(os.path.join(track_dir, '*.csv')))
		if len(track_list) == 1:
			break
		Parallel(n_jobs=-1)(delayed(link)(i, track_list) for i in range(0, (len(track_list) // 2) * 2, 2))
	untrimmed_path = os.path.join(scratch_dir, 'untrimmed_tracks.csv')
	os.system('mv %s %s' % (track_list[0], untrimmed_path))
	os.rmdir(track_dir)

def plot_untrimmed(scratch_dir):
	untrimmed_path = os.path.join(scratch_dir, 'untrimmed_tracks.csv')
	untrimmed_df = pd.read_csv(untrimmed_path)
	unique_values, counts = np.unique(untrimmed_df['id'], return_counts=True)
	plt.hist(counts, bins = np.arange(0, np.max(counts), 10)); plt.yscale('log')
	plt.xlabel('Length of track in frames'); plt.ylabel('Count'); plt.tight_layout()

def trim_csv(scratch_dir, min_frames_seen, min_area_traveled):
	untrimmed_path = os.path.join(scratch_dir, 'untrimmed_tracks.csv')
	untrimmed_df = pd.read_csv(untrimmed_path)
	trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
	unique_values, counts = np.unique(untrimmed_df['id'], return_counts=True)
	trimmed_df = untrimmed_df[np.isin(untrimmed_df['id'], unique_values[counts > min_frames_seen])]
	bbox_area = lambda xs, ys: (np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys))
	ids_to_save = []
	for i in sorted(set(trimmed_df['id'])):
		x, y = trimmed_df[trimmed_df['id'] == i]['x'], trimmed_df[trimmed_df['id'] == i]['y']
		if bbox_area(x, y) > min_area_traveled:
			ids_to_save.append(i)
	trimmed_df = trimmed_df[np.isin(trimmed_df['id'], ids_to_save)]
	ids = sorted(set(trimmed_df['id']))
	trimmed_df['id'] = trimmed_df['id'].replace(dict(zip(ids, range(len(ids)))))
	trimmed_df.to_csv(trimmed_path, index=False)

def plot_trimmed(scratch_dir):
	trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
	trimmed_df = pd.read_csv(trimmed_path)
	plt.subplot(211)
	unique_values, counts = np.unique(trimmed_df['frame'], return_counts=True)
	plt.plot(unique_values, counts)
	plt.ylabel('Worms per frame'); plt.xlabel('Frame #')
	plt.subplot(212)
	unique_values, counts = np.unique(trimmed_df['id'], return_counts=True)
	plt.hist(counts, bins = np.arange(0, np.max(counts), 10)); plt.yscale('log')
	plt.xlabel('Length of track in frames'); plt.ylabel('Count'); plt.tight_layout()
	plt.savefig(os.path.join(scratch_dir, 'tracking_stats.png'), dpi=300)

def plot_tracks(scratch_dir, mask):
	trimmed_path = os.path.join(scratch_dir, 'trimmed_tracks.csv')
	trimmed_df = pd.read_csv(trimmed_path)
	plt.figure(figsize=(5, 5))
	plt.imshow(mask, cmap='Greys_r')
	for i in trimmed_df['id']:
	    r = trimmed_df[trimmed_df['id'] == i].sort_values('frame')
	    plt.plot(r['x'], r['y'])
	plt.axis('off'); plt.tight_layout()
	plt.savefig(os.path.join(scratch_dir, 'tracks.png'), dpi=300)

def save_params(scratch_dir, sigma, thresh, area_range, ecc_range, seperation, min_frames_seen, min_area_traveled):
    index = ['SIGMA', 'THRESHOLD', 'AREA_RANGE', 'ECCENTRICITY_RANGE', 'SEPERATION', 'MIN_FRAMES_SEEN', 'MIN_AREA_TRAVELED']
    data = [sigma, thresh, area_range, ecc_range, seperation, min_frames_seen, min_area_traveled]
    df = pd.DataFrame(data, index=index)
    df.to_csv(os.path.join(scratch_dir, 'params.csv'), header=False)

def copy_to_output(input_avi, output_path, scratch_dir):
	exp_name = name_from_path(input_avi)
	output_dir = os.path.join(output_path, exp_name)
	os.makedirs(output_dir, exist_ok=True)
	os.system('cp %s/trimmed_tracks.csv %s/%s_trimmed_tracks.csv' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/params.csv %s/%s_params.csv' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/mask.tif %s/%s_mask.tif' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/tracks.png %s/%s_tracks.png' % (scratch_dir, output_dir, exp_name))
	os.system('cp %s/tracking_stats.png %s/%s_tracking_stats.png' % (scratch_dir, output_dir, exp_name))

