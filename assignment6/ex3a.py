from PIL import Image
import numpy as np
import os

# Rute
rute = 'assignment6/data/faces/1/'

# Matrix to add
all_columns_matrix = []

for i in range(1, 65): 
    image_name = f'{i:03d}.png' 
    image_rute = os.path.join(rute, image_name)

    image = Image.open(image_rute).convert('L')

    # Reshape into a column
    column = np.array(image).reshape(-1, 1)

    # Add
    all_columns_matrix.append(column)


final_matrix = np.hstack(all_columns_matrix)

print(final_matrix.shape)
