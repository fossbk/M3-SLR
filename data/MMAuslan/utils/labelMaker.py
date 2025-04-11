import pandas as pd
import json

def process_files(video_path_csv, word_label_csv, json_file, output_csv, output_center, output_left, output_right):
    """
    Process data from 3 input files and create output CSV files

    Args:
        video_path_csv: CSV file containing information about video paths
        word_label_csv: CSV file containing mapping from word to label_id
        json_file: JSON file containing mapping from ID to word
        output_csv: Output CSV file containing 5 columns: center, left, right, label_id, ID
        output_center: Output CSV file containing 2 columns: view (center), label_id
        output_left: Output CSV file containing 2 columns: view (left), label_id
        output_right: Output CSV file containing 2 columns: view (right), label_id
    """
    # Read the video path CSV file
    videos_df = pd.read_csv(video_path_csv)

    # Read the JSON file mapping ID to word
    with open(json_file, 'r', encoding='utf-8') as f:
        id_to_word = json.load(f)

    # Read the CSV file mapping word to label_id
    word_label_df = pd.read_csv(word_label_csv)

    # Debug: Check input data
    print("\n===== DEBUG: CHECKING INPUT DATA =====")
    print(f"Number of video paths: {len(videos_df)}")
    print(f"Number of IDs in JSON: {len(id_to_word)}")
    print(f"Number of words in word_label: {len(word_label_df)}")

    # Display some samples from each data source
    print("\nSample from videos_df:")
    print(videos_df.head(3))

    print("\nSample from id_to_word:")
    sample_ids = list(id_to_word.keys())[:3]
    for id_val in sample_ids: # Renamed 'id' to 'id_val' to avoid shadowing built-in
        print(f"{id_val}: {id_to_word[id_val]}")

    print("\nSample from word_label_df:")
    print(word_label_df.head(3))

    # Check ID format in data sources
    print("\nIDs in videos_df:")
    sample_video_ids = videos_df['ID'].astype(str).head(3).tolist() # Ensure ID is string for consistent type checking
    for id_val in sample_video_ids:
        print(f"'{id_val}' - Type: {type(id_val)}")

    print("\nIDs in id_to_word:")
    for id_val in sample_ids:
        print(f"'{id_val}' - Type: {type(id_val)}")

    # Create a dictionary from word to label_id
    word_to_label_id = dict(zip(word_label_df['word'], word_label_df['label_id']))

    # Create label_id column on videos_df - with checks and debugging
    def get_label_id(video_id):
      # Convert video_id to string and ensure 5-digit format with leading zeros
      video_id_str = str(video_id).zfill(5)

      # Try matching the ID directly (as 5-digit string)
      if video_id_str in id_to_word:
          word = id_to_word[video_id_str]
          if word in word_to_label_id:
              return word_to_label_id[word]

      # Try matching the ID after stripping leading zeros (if necessary)
      # Convert original video_id to string before stripping
      video_id_stripped = str(video_id).lstrip('0')
      if video_id_stripped in id_to_word:
          word = id_to_word[video_id_stripped]
          if word in word_to_label_id:
              return word_to_label_id[word]

      # Try matching the original ID as a string directly (in case it wasn't numeric initially)
      video_id_orig_str = str(video_id)
      if video_id_orig_str in id_to_word:
           word = id_to_word[video_id_orig_str]
           if word in word_to_label_id:
               return word_to_label_id[word]

      return None # Return None if no match found

    # Apply the function to create the label_id column
    # Ensure the 'ID' column is treated consistently, perhaps as string initially
    videos_df['label_id'] = videos_df['ID'].apply(get_label_id)

    # Check how many entries got a label_id
    print(f"\nNumber of entries with label_id after processing: {videos_df['label_id'].count()}")

    # Print some samples with label_id for checking
    print("\nSome samples with label_id:")
    print(videos_df[videos_df['label_id'].notnull()].head(3))

    # If still no IDs match, perform detailed checks
    if videos_df['label_id'].count() == 0:
        print("\n===== DETAILED DEBUGGING =====")
        # Check if any IDs from videos_df appear in id_to_word, considering different formats
        found_ids_details = []
        # Convert videos_df['ID'] to string for reliable comparison
        video_ids_str_unique = videos_df['ID'].astype(str).unique()
        for video_id_str in video_ids_str_unique:
            # Check original string form
            if video_id_str in id_to_word:
                found_ids_details.append(f"{video_id_str} (original)")
                continue # Found, no need to check others for this ID

            # Check padded form
            video_id_padded = video_id_str.zfill(5)
            if video_id_padded in id_to_word:
                found_ids_details.append(f"{video_id_str} (as {video_id_padded})")
                continue # Found

            # Check stripped form
            video_id_stripped = video_id_str.lstrip('0')
            # Avoid checking empty string if original was '0' or '00000'
            if video_id_stripped and video_id_stripped in id_to_word:
                found_ids_details.append(f"{video_id_str} (as {video_id_stripped})")
                continue # Found


        print(f"Number of unique video IDs from videos_df found in JSON (any format): {len(found_ids_details)}")
        if found_ids_details:
            print("Some IDs found (original format shown, matched format in parenthesis):")
            for id_detail in found_ids_details[:10]: # Show up to 10 matches
                print(f"- {id_detail}")
        else:
            print("No IDs from videos_df were found in the JSON keys using tested formats (original, 5-digit padded, leading-zero stripped).")

        # Check some specific IDs from both sources
        print("\nChecking some specific IDs:")
        video_ids_to_check = videos_df['ID'].astype(str).head(5).tolist()
        json_ids_to_check = list(id_to_word.keys())[:5]

        print("Sample IDs from videos_df (as strings):")
        for id_val in video_ids_to_check:
            print(f"- '{id_val}'")

        print("Sample IDs (keys) from JSON:")
        for id_val in json_ids_to_check:
            print(f"- '{id_val}'")

    # Filter out rows without a label_id
    filtered_df = videos_df.dropna(subset=['label_id'])

    # If no rows remain, print an error message and return
    if len(filtered_df) == 0:
        print("\n===== ERROR =====")
        print("No rows have a label_id after processing, cannot create output files.")
        print("Please check the input data format and ensure IDs match between video_path_csv and json_file.")
        return

    # Convert label_id to int type
    # Use .copy() to avoid SettingWithCopyWarning
    filtered_df = filtered_df.copy()
    filtered_df['label_id'] = filtered_df['label_id'].astype(int)

    # Select and rename columns for the main output file
    output_main_df = filtered_df[['center', 'left', 'right', 'label_id', 'ID']].copy()
    # Save the output DataFrame with 5 columns
    output_main_df.to_csv(output_csv, index=False)

    # Create and save the 3 sub-files
    center_df = pd.DataFrame({
        'view': filtered_df['center'],
        'label_id': filtered_df['label_id']
    })
    center_df.to_csv(output_center, index=False)

    left_df = pd.DataFrame({
        'view': filtered_df['left'],
        'label_id': filtered_df['label_id']
    })
    left_df.to_csv(output_left, index=False)

    right_df = pd.DataFrame({
        'view': filtered_df['right'],
        'label_id': filtered_df['label_id']
    })
    right_df.to_csv(output_right, index=False)

    # Print statistical information
    print(f"\n===== RESULTS =====")
    print(f"Total initial samples in video CSV: {len(videos_df)}")
    print(f"Number of samples after filtering (with label_id): {len(filtered_df)}")
    print(f"Main output file created at: {output_csv}")
    print(f"Center view file created at: {output_center}")
    print(f"Left view file created at: {output_left}")
    print(f"Right view file created at: {output_right}")


# Using the function
if __name__ == "__main__":
    # Paths to input files
    video_path_csv = "train_file.csv"
    word_label_csv = "lookuptable.csv"
    json_file = "Train.json"

    # Paths to output files
    output_csv = "threeview_train_labels.csv"
    output_center = "center_train_labels.csv"
    output_left = "left_train_labels.csv"
    output_right = "right_train_labels.csv"

    process_files(
        video_path_csv,
        word_label_csv,
        json_file,
        output_csv,
        output_center,
        output_left,
        output_right
    )
