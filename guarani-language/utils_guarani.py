
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements_Dataset(dataset, num_examples=10):
    '''
        Show 10 random sentences from the dataset.
    '''
    dataset = dataset.remove_columns(["path","audio"])
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))