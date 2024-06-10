def get_num_labels(dataset_name):
    num_labels = {
        "ag_news": 4,
        
        "sst2": 2,
        
    }
    
    
    return num_labels[dataset_name]