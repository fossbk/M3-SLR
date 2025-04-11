import numpy as np

def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad'],temporal_stride = 2):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    if train_p == 'fusion': 
        train_p = np.random.choice(['consecutive', 'random','segment','center_stride'])
    assert train_p in ['consecutive', 'random','segment','center_stride']
    assert train_m in ['pad']
    assert test_p in ['central', 'start', 'end','segment','center_stride']
    assert test_m in ['pad', 'start_pad', 'end_pad']
    if num_frames > 0:
        assert num_frames%4 == 0
        if is_train:
            if vlen > num_frames:
                
                if train_p == 'consecutive':
                    start = np.random.randint(0, vlen - num_frames, 1)[0]
                    selected_index = np.arange(start, start+num_frames)
                elif train_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif train_p == 'random':
                    # random sampling
                    selected_index = np.arange(vlen)
                    np.random.shuffle(selected_index)
                    selected_index = selected_index[:num_frames]  #to make the length equal to that of no drop
                    selected_index = sorted(selected_index)
                elif train_p == "segment":
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                else:
                    selected_index = np.arange(0, vlen)
            elif vlen < num_frames:
                if train_m == 'pad':
                    remain = num_frames - vlen
                    selected_index = np.arange(0, vlen)
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
            else:
                selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif test_p == 'start':
                    start = 0
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'end':
                    start = vlen - num_frames
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == "segment":
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                else: 
                    selected_index = np.arange(start, start+num_frames)
            else:
                remain = num_frames - vlen
                selected_index = np.arange(0, vlen)
                if test_m == 'pad':
                    pad_left = remain // 2
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'start_pad':
                    pad_left = 0
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'end_pad':
                    pad_left = remain
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
    else:
        # for statistics
        selected_index = np.arange(vlen)

    return selected_index, pad

def pad_index(index_arr, l_and_r) :
    left, right = l_and_r
    index_arr = index_arr.tolist()
    index_arr = left*[index_arr[0]] + index_arr + right*[index_arr[-1]]
    return np.array(index_arr)
    
