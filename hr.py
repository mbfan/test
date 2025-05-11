import pandas as pd
import os
import math

def calculate_a_b_from_scale(scale_dict):

  if not scale_dict:
    print("Hahahaha The reason for the error is that the input scale dictionary is empty.")
    return None, None
  try:
    values = list(scale_dict.values())
    if not all(isinstance(v, (int, float)) for v in values):
        print("Hahahaha The reason for the error is that the scale dictionary contains non-numeric values.")
        return None, None
  except Exception as e:
      print(f"Error processing scale value: {e}")
      return None, None
  if not values:
      print("The error occurs because the value cannot be found in the scale dictionary.")
      return None, None
  Hmax = max(values)
  Pmax_scale = Hmax * Hmax
  if Pmax_scale == 0:
    print("Since the value of Pmax_scale is zero, a and b cannot be calculated.")
    return None, None
  try:
    a = 4 / (Pmax_scale**2)
    b = 4 / Pmax_scale
  except ZeroDivisionError:
      print("The error occurs because a division by zero occurs when calculating a/b.")
      return None, None
  return a, b

# Read the specified hydrophilicity scale file here!! If you three want to run the code, remember to change the address here.
hydropathy_scale_path = r"C:\Jupyter Notebook\UOB labs\DSMP\Sapienza_data\progetto_bristol\hydropathy\ENGD860101.csv"

if not os.path.exists(hydropathy_scale_path):
    print(f"The reason for the error is that the specified scale file does not exist: {hydropathy_scale_path}")
    exit(1)

try:
    hydropathy_scale_df = pd.read_csv(hydropathy_scale_path, header=None, names=['Residue', 'Hydropathy_Value'])
    print(f"Read the scale file: {hydropathy_scale_path}")
except Exception as e:
    print(f"An error occurred while reading the scale file {hydropathy_scale_path}: {e}")
    exit(1)

hydropathy_dict = dict(zip(hydropathy_scale_df['Residue'], hydropathy_scale_df['Hydropathy_Value']))

# Start calculating the values ​​of a and b here. There is a detailed description in the teacher’s PPT, or you can ask me.
a_calculated, b_calculated = calculate_a_b_from_scale(hydropathy_dict)

if a_calculated is None or b_calculated is None:
    print("The error is caused by failure to calculate a and b values ​​from the scale file")
    exit(1)
else:
    print(f"According to the scale file {os.path.basename(hydropathy_scale_path)} :")
    print(f"  a = {a_calculated:.8f}")
    print(f"  b = {b_calculated:.8f}")

# This step reads the data set. Remember to change the path for different data sets.
new_dataset_path = r"C:\Jupyter Notebook\UOB labs\DSMP\Sapienza_data\progetto_bristol\dataset\test\dataset.txt"

if os.path.exists(new_dataset_path):
    try:
        data = pd.read_csv(new_dataset_path, sep=',')
        print(f"Successfully read the dataset file: {new_dataset_path}")
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        exit(1)
else:
    print(f"Error: Dataset file does not exist: {new_dataset_path}")
    exit(1)

results_hr = []

# There was a problem here before, and I finally found the reason. I hope you can pay attention to it.
res_a_col_name = 'resA' 
res_b_start_index = 2   
res_b_end_index = 11    

if res_a_col_name not in data.columns:
    print(f"The reason for the error: The dataset file is missing column '{res_a_col_name}' .")
    exit(1)
if res_b_end_index >= len(data.columns):
     print(f"The reason for the error is the number of columns in the dataset file ({len(data.columns)}) not enough to cover the index {res_b_end_index}。")
     print(f"  Available columns: {list(data.columns)}")
     exit(1)

print(f"Start calculating Hr value (processing column index {res_b_start_index} to {res_b_end_index})...")
for index, row in data.iterrows():
    residue_A = row[res_a_col_name]

    H_A = hydropathy_dict.get(residue_A, 0.0) 

    hr_values_for_row = []

    for i in range(res_b_start_index, res_b_end_index + 1):
        try:
            residue_B = row.iloc[i]
            H_B = hydropathy_dict.get(residue_B, 0.0) 

            product_HAB = H_A * H_B
            H_r = -a_calculated * (product_HAB ** 2) + b_calculated * product_HAB
            hr_values_for_row.append(H_r)
        except IndexError:
             print(f"Row {index} Error accessing column index {i}. Added default value 0.0.")
             hr_values_for_row.append(0.0)
        except Exception as e:
             print(f"Warning: Error processing row {index} column index {i}: {e}. Added default value 0.0.")
             hr_values_for_row.append(0.0)

    if len(hr_values_for_row) != 10:
        print(f"Row {index} generated {len(hr_values_for_row)} Hr values, 10 expected.")

    hr_values_str = ' '.join(map(lambda x: f"{x:.8f}", hr_values_for_row))
    results_hr.append(hr_values_str)

print("The Hr value calculation is completed.")

output_content = '\n'.join(results_hr)
# Here you three have to define the output file path yourself.
output_file_path = r"C:\Jupyter Notebook\UOB labs\DSMP\Sapienza_data\progetto_bristol\hr files\test\hr_ENGD860101.txt"

try:
    with open(output_file_path, 'w') as file:
        file.write(output_content)
    print(f"Results saved successfully to: {output_file_path}")
except Exception as e:
    print(f"Error writing output file: {e}")

print("The script has finished executing.")