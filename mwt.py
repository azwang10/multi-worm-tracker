import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, morphology, measure
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
plt.rcParams['font.sans-serif'] = 'Arial'

name_from_path = lambda name: os.path.splitext(os.path.basename(name))[0]

def make_scratch_dir(input_avi, scratch_path):
	if not os.path.isfile(input_avi):
		raise FileNotFoundError()
	exp_name = name_from_path(input_avi)
	scratch_dir = os.path.join(scratch_path, exp_name + '-scratch/')
	os.makedirs(scratch_dir, exist_ok=True)
	return scratch_dir

def make_mask(input_avi, scratch_dir):
	mask_path = scratch_dir + 'mask.tif'
	if os.path.isfile(mask_path):
		mask = io.imread(mask_path)
	else:
		minute_dir = scratch_dir + 'minute/'
		os.makedirs(minute_dir, exist_ok=True)

		#extracts a frame at each minute interval from the recording and saves it to minute_dir
		os.system(f"ffmpeg -hide_banner -i '{input_avi}' -vf select='not(mod(n\,180))' -fps_mode vfr -pix_fmt gray '{minute_dir}%06d.tif'")
		minute_arr = np.array([io.imread(i) for i in sorted(glob(f'{minute_dir}*.tif'))])
		mask = np.median(minute_arr, axis=0).astype(np.ubyte)
		io.imsave(mask_path, mask)
	plt.imshow(mask, cmap='Greys_r', vmin=0, vmax=255)
	return mask

def guess_threshold(scratch_dir, mask, threshold_guesses=range(2, 11), size_range=(50, 150)):

	#subtracts each frame from the mask for every minute and reshapes the stack to a single row
	minute_paths = sorted(glob(scratch_dir + 'minute/*.tif'))
	minute_stack = np.array([io.imread(minute_path) for minute_path in minute_paths])
	mask_stack = np.tile(mask, (len(minute_paths), 1, 1))
	diff_stack = mask_stack.astype(np.int16) - minute_stack.astype(np.int16)
	diff_stack = np.clip(diff_stack, 0, 255).astype(np.ubyte)
	long_stack = diff_stack.reshape(-1, diff_stack.shape[-1])

	#a function to quickly get the size for each blob, filtering out blobs less than the size_range minimum
	def get_sizes(threshold):
		thresholded_stack = morphology.remove_small_objects(long_stack > threshold, min_size=size_range[0])
		labeled_stack = measure.label(thresholded_stack)
		return np.bincount(labeled_stack[labeled_stack > 0])

	#gets the threshold that produces blobs closest to median_size
	desired_size = np.mean(size_range)
	minute_sizes = np.array([np.median(get_sizes(threshold)) for threshold in threshold_guesses])
	best_threshold = threshold_guesses[np.argmin(abs(minute_sizes - desired_size))]

	#plots the histogram of sizes
	sizes = get_sizes(best_threshold)
	plt.title(f'Best Threshold = {best_threshold}')
	plt.hist(sizes, bins=np.arange(0, 200))
	plt.axvline(desired_size, c='k')
	plt.axvline(size_range[0], c='k', linestyle='--'); plt.axvline(size_range[1], c='k', linestyle='--')
	plt.ylabel('# of worms'); plt.xlabel('Sizes (px)')
	return best_threshold

def convert_to_tif(input_avi, scratch_dir):

	#extracts every frame from the recording and saves it to tif_dir
	tif_dir = scratch_dir + 'tif/'
	os.makedirs(tif_dir, exist_ok=True)
	os.system(f"ffmpeg -hide_banner -i '{input_avi}' -pix_fmt gray '{tif_dir}%06d.tif'")
	print(f"Frames converted: {len(glob(tif_dir + '*.tif'))}")

def detect_worms(scratch_dir, mask, best_threshold, size_range=(50, 150), ecc_range=(0.5, 1)):
	worm_dir = scratch_dir + 'worms/'
	os.makedirs(worm_dir, exist_ok=True)

	#for each tif file, get_worms() subtracts the mask, thresholds, and finds worms in the right size range and saves their positions
	def get_worms(tif_path):
		num_str = name_from_path(tif_path)
		worm_path = worm_dir + num_str + '.csv'
		if os.path.isfile(worm_path):
			return
		diff_frame = mask.astype(np.int16) - io.imread(tif_path).astype(np.int16)
		diff_frame = np.clip(diff_frame, 0, 255).astype(np.ubyte)
		thresholded_frame = morphology.remove_small_objects(diff_frame > best_threshold, min_size=size_range[0])
		labeled_frame = measure.label(thresholded_frame)
		worm_df = pd.DataFrame(measure.regionprops_table(labeled_frame, properties=['area', 'centroid', 'eccentricity']))
		worm_indexes = np.all([worm_df['area'] > size_range[0], worm_df['area'] < size_range[1],
							   worm_df['eccentricity'] > ecc_range[0], worm_df['eccentricity'] < ecc_range[1]], axis=0)
		worm_df = worm_df.iloc[worm_indexes]

		#makes a new dataframe output that rounds positions and adds the frame number and worm number
		out = pd.DataFrame()
		out['frame'] = [int(num_str)] * len(worm_df)
		out[['x', 'y']] = worm_df[['centroid-1', 'centroid-0']].round(2).to_numpy()
		out['id'] = range(len(out))
		out.to_csv(worm_path, index=False)

	#runs in parallel for each tif file
	tif_dir = scratch_dir + 'tif/'
	tif_list = sorted(glob(tif_dir + '*.tif'))
	Parallel(n_jobs=-1)(delayed(get_worms)(i) for i in tif_list)

	#plots the worms
	worm_list = sorted(glob(worm_dir + '*.csv'))
	plt.plot([len(pd.read_csv(i)) for i in worm_list])
	plt.ylabel('Worms per frame'); plt.xlabel('Frame #')
	plt.title('Untracked worms')

def link_worms(scratch_dir, max_speed=0.6):
	untrimmed_path = scratch_dir + 'untrimmed_tracks.csv'
	if os.path.isfile(untrimmed_path):
		pass
	else:
		worm_dir = scratch_dir + 'worms/'
		os.system(f'cd {worm_dir} && zip -0 -r -q {scratch_dir}worms.zip *.csv')
		track_dir = scratch_dir + 'tracks/'
		os.makedirs(track_dir, exist_ok=True)
		os.system(f'unzip -o -q {scratch_dir}worms.zip -d {track_dir}')

		#function to link two consecutive dfs in the track directory
		seperation = max_speed / 3 * 24 #(speed mm/s) / (3 frame/s) * (24 frames/mm) 
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

		#runs link until only one df is left
		while True:
			track_list = sorted(glob(track_dir + '*.csv'))
			if len(track_list) == 1:
				break
			Parallel(n_jobs=-1)(delayed(link)(i, track_list) for i in range(0, (len(track_list) // 2) * 2, 2))
		os.system(f'mv {track_list[0]} {untrimmed_path}')
		os.rmdir(track_dir)

	#plots all untrimmed tracks
	untrimmed_df = pd.read_csv(untrimmed_path)
	unique_values, counts = np.unique(untrimmed_df['id'], return_counts=True)
	plt.hist(counts, bins=np.arange(0, np.max(counts), 10)); plt.yscale('log')
	plt.xlabel('Length of track in frames'); plt.ylabel('Count')
	plt.title('Tracked worms')

def trim_csv(scratch_dir, min_secs_tracked=30, min_area_tracked=2):
	trimmed_path = scratch_dir + 'trimmed_tracks.csv'
	if os.path.isfile(trimmed_path):
		trimmed_df = pd.read_csv(trimmed_path)
	else:
	
		#trims the df to only tracks that have been seen for more than min_secs_tracked
		min_frames_tracked = 3 * min_secs_tracked
		untrimmed_df = pd.read_csv(scratch_dir + 'untrimmed_tracks.csv')
		unique_values, counts = np.unique(untrimmed_df['id'], return_counts=True)
		trimmed_df = untrimmed_df[np.isin(untrimmed_df['id'], unique_values[counts > min_frames_tracked])]

		#trims the df to only tracks that have traveled more than the min_area_tracked
		min_px_tracked = min_area_tracked * 24 * 24 #conversion from mm^2 to px^2
		ids_to_save = []
		for i in sorted(set(trimmed_df['id'])):
			pts = trimmed_df[trimmed_df['id'] == i][['x', 'y']].to_numpy()
			if ConvexHull(pts).volume > min_px_tracked:
				ids_to_save.append(i)
		trimmed_df = trimmed_df[np.isin(trimmed_df['id'], ids_to_save)]
		ids = sorted(set(trimmed_df['id']))
		trimmed_df['id'] = trimmed_df['id'].replace(dict(zip(ids, range(len(ids)))))
		trimmed_df.to_csv(trimmed_path, index=False)
	
	#plots the tracks
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
	trimmed_df = pd.read_csv(scratch_dir + 'trimmed_tracks.csv')
	plt.figure(figsize=(5, 5))
	plt.imshow(mask, cmap='Greys_r')
	plt.scatter(trimmed_df['x'], trimmed_df['y'], c=trimmed_df['id'] % 20, cmap='tab20', s=0.5, lw=0)
	plt.axis('off'); plt.tight_layout()
	plt.savefig(scratch_dir + 'tracks.png', dpi=300)

def save_output(input_avi, scratch_dir, output_path, best_threshold, size_range, ecc_range, max_speed, min_secs_tracked, min_area_tracked):

	#makes a .txt of all the parameters used during analysis
	index = ['THRESHOLD', 'SIZE_RANGE', 'ECCENTRICITY_RANGE', 'MAX_SPEED', 'MIN_SECS_TRACKED', 'MIN_AREA_TRACKED']
	data = [best_threshold, size_range, ecc_range, max_speed, min_secs_tracked, min_area_tracked]
	df = pd.DataFrame(data, index=index)
	df.to_csv(os.path.join(scratch_dir, 'params.txt'), header=False, sep='\t')

	#makes the output directory and copies all files to it
	exp_name = name_from_path(input_avi)
	output_dir = os.path.join(output_path, exp_name + '/')
	os.makedirs(output_dir, exist_ok=True)
	os.system(f'cp {scratch_dir}trimmed_tracks.csv {output_dir}{exp_name}_trimmed_tracks.csv')
	os.system(f'cp {scratch_dir}params.txt {output_dir}{exp_name}_params.txt')
	os.system(f'cp {scratch_dir}mask.tif {output_dir}{exp_name}_mask.tif')
	os.system(f'cp {scratch_dir}tracking_stats.png {output_dir}{exp_name}_tracking_stats.png')
	os.system(f'cp {scratch_dir}tracks.png {output_dir}{exp_name}_tracks.png')

