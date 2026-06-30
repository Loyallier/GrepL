GrepL: AI-Powered Campus Lost and Found
========================================

1. Overview
-----------

GrepL helps users search for lost items in a registered collection of found
items. The user describes a lost item and may add a date range or campus
location. The program compares the description with found-item images, ranks
the candidates, and displays the most likely matches in a browser interface.

The matching process uses TensorFlow and TF-CLIP for text-to-image comparison.
Time and location filters are also considered when the final results are
ranked.


2. Requirements
---------------

- Python 3.10
- A modern web browser
- Internet access while installing the Python packages
- Internet access on the first search if the TF-CLIP model weights are not
  already available locally
- All project files kept in their original folder structure

The required Python packages are listed in requirements.txt. TensorFlow and
the model files require several gigabytes of free disk space.


3. Installation
---------------

Open a terminal in the project folder, which is the folder containing main.py.
Then install the dependencies:

    python -m pip install -r requirements.txt


4. Running the Program
----------------------

From the project folder, run:

    python main.py

The GrepL interface should open automatically in the default browser. If it
does not, open the following address manually:

    http://127.0.0.1:5000

Keep the terminal open while using GrepL. To stop the program, return to the
terminal and press Ctrl+C.


5. Using GrepL
--------------

1. Enter a clear description in the "Describe the item you lost" field.
   Include visible details such as the item type, colour, material, pattern,
   logo, or distinguishing marks.
2. Select the filter button if a lost date range, campus location, or result
   limit would help narrow the search.
3. Select Search.
4. If GrepL asks a clarification question, select the most suitable answer.
5. Review the ranked candidate cards. Each card shows the found location,
   found time, match score, and confidence level.
6. Select Details to view a larger image and the reasons for the match. The
   image can be zoomed and dragged.
7. Use the reset button to clear the current search and begin again.

Example description:

    Black water bottle with a white sticker and a silver lid

The description is required. Date and location filters are optional. A search
returns up to 10 candidate matches, depending on the selected result limit.


6. Preparing Found-Item Data (Optional)
---------------------------------------

The supplied data can be searched immediately when the submission folder
contains the registered records and their corresponding cropped images. Use
the following steps only when adding new found-item photographs.

1. Place the photographs in a separate folder. Supported formats are JPG,
   JPEG, PNG, WEBP, and BMP. Only files directly inside that folder are read;
   subfolders are not scanned.
2. Import the photographs:

       python scripts/add_raw_found_images.py "path/to/image_folder(not the image itself)"

3. For each photograph, enter the found date in YYYY-MM-DD format, the found
   hour from 0 to 23, and one of the location keys displayed in the terminal.
   Leave the date or hour empty if it is unknown.
4. Detect, crop, and register the found items:

       python scripts/run_registration.py

The first registration may take longer because the AI models must be loaded.
The original photographs are copied into data/raw_found_images, detected item
images are written to data/cropped_item_image, and searchable records are
stored under data/generated.


7. Main Project Files
---------------------

- main.py: starts the GrepL browser application.
- requirements.txt: lists the required Python packages.
- src/: contains the interface, search, ranking, detection, and AI modules.
- scripts/: contains the optional data import and registration commands.
- data/: contains found-item records, images, and generated embeddings.
- models/: contains the object-detection model files.


8. Troubleshooting
------------------

- "NiceGUI is not installed": run the installation command in Section 3.
- The browser does not open: visit http://127.0.0.1:5000 manually.
- The first search is slow: allow time for TensorFlow to load and for TF-CLIP
  weights to download. Do not close the terminal.
- No candidates are shown: make sure data/generated/found_items.json and the
  corresponding files in data/cropped_item_image are present. If it's still 
  unable to work, kindly refer to Section 6.
- Candidate images are unavailable: keep the data folder unchanged and confirm
  that the registered image files were included with the project.
- The application reports that the port is in use: close the other program
  using port 5000 and run python main.py again.

9. Team Members
------------------
SWE2409044  Pan Quanyou (Leader)
SWE2409015  Hong Guangyu
SWE2409030  Liang Yuqi
SWE2409038  Liu Yuanyuan
SWE2409046  Song Langkun
SWE2409061  Yan Nachuan