import numpy as np

def unwrap_with_nans(phases, unit='deg'):
    ''' Adaptation of the numpy.unwrap function to handling multiple consecutive NaNs, from ChatGPT.
    input: phases - 1D array of phase angles in degrees or radians, as specified by unit; unit - default deg
    output: 1D array of unwrapped phase angles in degrees or radians
    '''
    # Create a boolean mask for NaNs
    nan_mask = np.isnan(phases)
    
    # Split the array into contiguous non-NaN segments
    segments = []
    current_segment = []
    for i, val in enumerate(phases):
        if not np.isnan(val):
            current_segment.append(val)
        else:
            if current_segment:
                segments.append(np.array(current_segment))
                current_segment = []
            segments.append(np.array([np.nan]))
    if current_segment:
        segments.append(np.array(current_segment))
    
    # Apply unwrap to non-NaN segments
    unwrapped_segments = []
    for segment in segments:
        if np.isnan(segment).all():  # Skip the NaN segments
            unwrapped_segments.append(segment)
        else:
            # Convert degrees to radians for correct unwrapping, then back to degrees
            if unit == 'deg':
                unwrapped_segment = np.unwrap(np.deg2rad(segment))
                unwrapped_segments.append(np.rad2deg(unwrapped_segment))
            else:
                unwrapped_segments.append(np.unwrap(segment))
    
    # Merge the unwrapped segments back into one array
    unwrapped_phases = np.concatenate(unwrapped_segments)
    
    return unwrapped_phases
