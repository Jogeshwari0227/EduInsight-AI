import pandas as pd

def preprocessed_data(df):
    # 1. Convert Target Class: L -> 0, M -> 1, H -> 2 [cite: 4]
    class_map = {'L': 0, 'M': 1, 'H': 2}
    if 'Class' in df.columns:
        df['Class'] = df['Class'].map(class_map)
    
    # 2. Convert StudentAbsenceDays: Under-7 -> 0, Above-7 -> 1 [cite: 4]
    absence_map = {'Under-7': 0, 'Above-7': 1}
    if 'StudentAbsenceDays' in df.columns:
        df['StudentAbsenceDays'] = df['StudentAbsenceDays'].map(absence_map)
    
    # 3. Select the core behavioral features [cite: 4]
    features = [
        'raisedhands', 
        'VisITedResources', 
        'AnnouncementsView', 
        'Discussion', 
        'StudentAbsenceDays'
    ]

    # If this is training data, include the target Class
    if 'Class' in df.columns:
        return df[features], df['Class']
    
    # If this is for prediction, just return features
    return df[features]