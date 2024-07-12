import torch
import pandas as pd
import numpy as np

from inception import InceptionModel


PATH = "C:/Users/Computer/Desktop/Project Tech/API_streamlit/Code_Quan/model_99"

model = InceptionModel(num_blocks=4,
            in_channels=64,
            out_channels=[128, 256, 256, 256],
            bottleneck_channels=[32,64,64,64],
            kernel_sizes=45,
            use_residuals=[True,True,True, False],
            num_pred_classes=1)

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

def segment_recording2(recording, segment_length=2000):
    """Segment a recording into 10-second chunks."""
    # num_segments = recording.shape[0] // segment_length
    segments = []
    for i in range(0,len(recording) - segment_length + 1, 500):
      segment = recording[i:i+segment_length]
      segments.append(segment)
    num_segments = len(segments)
    return segments, num_segments

def return_rate(df): 
    # df = pd.read_csv(df, usecols=[25], header=None)

    csi_data = df[25].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))
    csi_data = np.stack(csi_data.values)

    csi_data_list = []

    segmented_data, num_segments = segment_recording2(csi_data)
    csi_data_list.append(segmented_data)
    csi_data_list = np.concatenate(csi_data_list, axis=0)

    A = np.sqrt(csi_data_list[:,:,::2]**2+csi_data_list[:,:,1::2]**2)
    A = np.transpose(A,(0,2,1))
    A = torch.FloatTensor(A)
    
    mean_value = (60 * model(A).squeeze()).mean().item()

    return mean_value  