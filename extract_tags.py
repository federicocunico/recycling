import json
import os
from typing import Dict, OrderedDict
import cv2
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path

os.environ["TESSDATA_PREFIX"] = os.path.join(
    os.getcwd(), "ocr"
)  # directory for tesseract pre-trained data

from pytesseract import Output
import pytesseract

from shared import *


# DEBUG CONFIGS
winname = "window"
DEBUG = False

# PDF CONFIGS
curr_file = os.path.join("data", "2021.pdf")


# OCR CONFIGS
# docs:
# Page segmentation modes (PSM):
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR.
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#                         bypassing hacks that are Tesseract-specific.

# OCR Engine modes (OEM):
# 0. Legacy engine only.
# 1. Neural nets LSTM engine only.
# 2. Legacy + LSTM engines.
# 3. Default, based on what is available.

day_ocr_config = r"--oem 3 --psm 3 --dpi 300"
waste_ocr_config = r"--oem 3 --psm 7 --dpi 300"
find_most_ocr_config = r"--oem 3 --psm 11 --dpi 300"
min_conf = 0


def extract_pages(pdf_file: str, dst: str):

    if not os.path.isfile(pdf_file):
        raise FileNotFoundError(pdf_file)

    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    if len(os.listdir(dst)) > 1:
        return

    pages = convert_from_path(pdf_file, 500)
    idx = 0
    for i, page in tqdm(enumerate(pages)):
        if i in [0, 1]:
            continue

        # try:
        #     m = months[str(idx)]
        # except:
        #     m = i
        m = i
        _ = os.path.join(dst, f"{m}.jpg")
        page.save(_, "JPEG")
        idx += 1


def get_files_from(directory: str):
    return sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ],
        key=len,
    )


def extract_crops(src: str, dst: str, force=False):

    images = get_files_from(src)

    os.makedirs(dst, exist_ok=True)

    # gennaio - dicembre
    images = images[-12:]

    # For each day; measures in px
    # max 40; sono 31 giorni massimo, ma ogni pagina ha uno spazio vuoto. Se ne hai di più?
    day_max_count = 33

    day_starting_topleft = 413
    day_vertical_step = 226

    fixed_left_start = 136
    fixed_stop_right = 2500

    for j, path in enumerate(images):
        img = Image.open(path)
        img = np.asarray(img)

        # b, _ = os.path.splitext(path)
        # month = b.split(os.sep)[-1]
        new_folder = os.path.join(dst, str(j))
        os.makedirs(new_folder, exist_ok=True)

        top = None
        bottom = None

        for day in range(0, day_max_count):

            f = os.path.join(new_folder, f"{day}.jpg")
            if os.path.exists(f) and not force:
                continue

            if top is None:
                top = day_starting_topleft

            bottom = top + day_vertical_step

            print("righe: da ", top, " a ", bottom)
            crop = img[top:bottom, fixed_left_start:fixed_stop_right, :]
            if 0 in crop.shape:
                print("Reached End of Image")
                break
            top = bottom

            cv2.imshow(
                winname, cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2))
            )
            q = cv2.waitKey(1)
            if q == ord("q"):
                break

            # COLOR
            # cv2.imwrite(f, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            # GRAY
            cv2.imwrite(f, cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
        if DEBUG:
            break


def extract_words_old(crop_dir: str, words_dir: str):

    data = {}

    directories = sorted(os.listdir(crop_dir), key=len)
    for d in directories:

        month = months[d]
        num_days = days_by_months[d]

        data[month] = {}

        dir = os.path.join(crop_dir, d)
        crops = get_files_from(dir)

        # IL GIORNO 17 c'è uno spazio bianco, così pare almeno, quindi...
        # assumendo siano ordinati...
        crops = crops[0:16] + crops[17:]

        day_num = 0
        for _, crop in enumerate(crops):

            gray_image = np.asarray(Image.open(crop))
            threshold_img = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )[1]

            cv2.imshow(
                winname,
                cv2.resize(
                    threshold_img,
                    (threshold_img.shape[1] // 2, threshold_img.shape[0] // 2),
                ),
            )
            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            # perform OCR
            try:
                details = pytesseract.image_to_data(
                    threshold_img,
                    output_type=Output.DICT,
                    config=day_ocr_config,
                    lang="ita",
                )
            except Exception as e:
                print("Unable to extract some data!", e)
                continue

            # print(details)

            img = threshold_img

            # Get detection details
            texts = details["text"]
            confs = details["conf"]
            bboxes_top_coords = details["top"]
            bboxes_left_coords = details["left"]
            bboxes_width = details["width"]
            bboxes_height = details["height"]

            # Filter by confidence
            if max([int(c) for c in confs]) < 0:
                # blank image, or no data found, continue
                continue

            found_day = None
            found_raccolta = []

            # For each detected text
            for i in range(len(texts)):
                highlight = False

                text = texts[i]
                conf = confs[i]
                try:
                    conf = int(conf)
                except:
                    print(
                        f'Unable to get confidence as integer, skipping. Value: "{conf}"'
                    )
                    conf = 0
                if conf < min_conf:
                    continue

                # try:
                #     day_n = int(text)
                #     print('Found day: ', day_n, ' from ', text)
                #     highlight= True
                # except:
                #     print('Day not found')

                # Check if text is a day
                for d in weekdays:
                    if d in text.lower():
                        # print('Found day: ', d, ' from ', text)
                        highlight = True
                        found_day = d
                        break

                # Check if text is one of the requested keywords
                raccolta = text.lower() in keywords
                if not raccolta and not highlight:
                    continue

                if raccolta:
                    found_raccolta.append(text.lower())

                x = bboxes_left_coords[i]
                y = bboxes_top_coords[i]
                w = bboxes_width[i]
                h = bboxes_height[i]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            day_num += 1

            data[month][day_num] = {
                "giorno": found_day,
                "raccolta": ",".join(found_raccolta),
            }
            print(
                month,
                "Giorno: ",
                day_num,
                found_day,
                " passano a raccogliere: ",
                found_raccolta,
            )

            cv2.imshow(winname, cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            k = cv2.waitKey(1)
            if k == ord("q"):
                break

            if day_num >= num_days:
                print("End of month")
                break

        if DEBUG:
            break

    return data


def _process_frame(crop: np.ndarray, ocr_conf, pre_process: bool = False):
    crop = crop.copy()

    if pre_process:
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform OCR
    try:
        details = pytesseract.image_to_data(
            crop, output_type=Output.DICT, config=ocr_conf, lang="ita"
        )
    except Exception as e:
        print("Unable to extract some data!", e)
        return None

    # Get detection details
    # texts = details["text"]
    confs = details["conf"]
    # bboxes_top_coords = details["top"]
    # bboxes_left_coords = details["left"]
    # bboxes_width = details["width"]
    # bboxes_height = details["height"]

    # Filter by confidence
    if max([float(c) for c in confs]) < 0.0:
        # blank image, or no data found, continue
        return None

    return details


def _get_day(day_crop, conf):
    details = _process_frame(day_crop, conf, pre_process=True)
    if not details:
        return None

    # Get detection details
    texts = details["text"]
    confs = details["conf"]
    bboxes_top_coords = details["top"]
    bboxes_left_coords = details["left"]
    bboxes_width = details["width"]
    bboxes_height = details["height"]

    found_day = None

    # For each detected text
    for i in range(len(texts)):
        highlight = False

        text = texts[i]
        conf = confs[i]
        try:
            conf = int(conf)
        except:
            print(f'Unable to get confidence as integer, skipping. Value: "{conf}"')
            conf = 0
        if conf < min_conf:
            continue

        # Check if text is a day
        for d in weekdays:
            if d in text.lower():
                # print('Found day: ', d, ' from ', text)
                highlight = True
                found_day = d
                break

        if highlight:
            x = bboxes_left_coords[i]
            y = bboxes_top_coords[i]
            w = bboxes_width[i]
            h = bboxes_height[i]

            cv2.rectangle(day_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return found_day

    return None


def _get_waste(waste_crop, conf):
    details = _process_frame(waste_crop, conf, pre_process=True)
    if not details:
        return []

    # Get detection details
    texts = details["text"]
    confs = details["conf"]
    bboxes_top_coords = details["top"]
    bboxes_left_coords = details["left"]
    bboxes_width = details["width"]
    bboxes_height = details["height"]

    found_raccolta = []

    # For each detected text
    for i in range(len(texts)):

        text = texts[i].lower()
        conf = confs[i]
        try:
            conf = int(conf)
        except:
            print(f'Unable to get confidence as integer, skipping. Value: "{conf}"')
            conf = 0
        if conf < min_conf:
            continue

        # Check if text is one of the requested keywords
        raccolta = text in keywords
        if not raccolta:
            continue

        if raccolta:
            found_raccolta.append(text)
            x = bboxes_left_coords[i]
            y = bboxes_top_coords[i]
            w = bboxes_width[i]
            h = bboxes_height[i]

            cv2.rectangle(waste_crop, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return found_raccolta


def extract_words(crop_dir: str, visualize: bool = False):

    day_region = 720  # 950  # up to
    # waste_region = 1800  # up to

    data = {}

    directories = sorted(os.listdir(crop_dir), key=len)
    for d in directories:

        # if d != "2":
        #     continue

        month = months[d]
        num_days = days_by_months[d]

        data[month] = {}

        dir = os.path.join(crop_dir, d)
        crops = get_files_from(dir)

        # IL GIORNO 17 c'è uno spazio bianco, così pare almeno, quindi...
        # assumendo siano ordinati...
        crops = crops[0:16] + crops[17:]

        day_num = 0
        for _, crop in enumerate(crops):

            gray_image = np.asarray(Image.open(crop))

            day_crop = gray_image[:, 0:day_region]
            # waste_crop = gray_image[:, day_region:waste_region]
            waste_crop = gray_image[:, day_region:]

            found_day = _get_day(day_crop, day_ocr_config)
            found_raccolta = _get_waste(waste_crop, waste_ocr_config)
            if not found_raccolta:
                # check more accurately
                # check = waste_crop[waste_crop<255]
                found_raccolta = _get_waste(gray_image, find_most_ocr_config)
                if found_raccolta:
                    print("Found with fallback:  ", end="")

            day_num += 1
            data[month][day_num] = {
                "giorno": found_day,
                "raccolta": ",".join(found_raccolta),
            }
            print(
                month,
                "Giorno: ",
                day_num,
                found_day,
                " passano a raccogliere: ",
                found_raccolta,
            )

            if visualize:
                cv2.imshow(
                    winname,
                    cv2.resize(
                        gray_image, (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
                    ),
                )
                cv2.imshow(
                    "day",
                    cv2.resize(day_crop, (day_crop.shape[1] // 2, day_crop.shape[0] // 2)),
                )
                cv2.imshow(
                    "waste",
                    cv2.resize(
                        waste_crop, (waste_crop.shape[1] // 2, waste_crop.shape[0] // 2)
                    ),
                )
                k = cv2.waitKey(50)
                if k == ord("q"):
                    break

            if day_num >= num_days:
                print("End of month")
                break

        if DEBUG:
            break

    return data


def validate(start_data: Dict):
    __day_str__ = "giorno"
    __raccolta_str__ = "raccolta"

    # data = OrderedDict(start_data)
    # # sequence = ["luned", "marted", "mercoled", "gioved", "venerd", "sabato", "domenica"]
    # for i, month_name in enumerate(data):
    #     month_data = data[month_name]

    #     a = None
    #     b = None
    #     for j, day_id in enumerate(month_data):
    #         day_data = month_data[day_id]
    #         day_name = day_data[__day_str__]
    #         raccolta = day_data[__raccolta_str__]
    #         if a is None:
    #             if day_name is None:
    #                 # first day is none, look for the next one
    #                 new_day = month_data[str(j+1)][__day_str__]
    #                 k = j
    #                 while new_day is None:
    #                     k += 1
    #                     next_day = month_data[str(k+1)][__day_str__]
    #                 diff = k - j  # number of days of difference
    #                 print("Diff: ", diff)
    #             else:
    #                 a = day_name

    def find_closest_not_null(_list, _starting_idx):
        v_a = None
        a = None
        # To the end
        for _i in range(_starting_idx, len(_list)):
            k, v = _list[_i]
            val = v[__day_str__]
            if val is not None:
                a = _i
                v_a = val
                break

        v_b = None
        b = None
        # To the beginning
        for _i in reversed(range(0, _starting_idx)):
            k, v = _list[_i]
            val = v[__day_str__]
            if val is not None:
                b = _i
                v_b = val
                break
        if a and b:
            d1 = abs(a - _starting_idx)
            d2 = abs(b - _starting_idx)
            if d1 <= d2:
                return a, v_a
            return b, v_b
        if not a:
            return b, v_b
        else:
            return b, v_b

    def next_forward(val, r):
        i = weekdays_forward_search.index(val)
        return weekdays_forward_search[(i + r) % len(weekdays_forward_search)]

    def next_backward(val, r):
        i = weekdays_backward_search.index(val)
        return weekdays_backward_search[(i + r) % len(weekdays_backward_search)]

    for i, month_name in enumerate(start_data):
        month_data = start_data[month_name]
        t = [(k, v) for k, v in month_data.items()]
        for i in range(len(t)):
            k, v = t[i]
            day = v[__day_str__]
            if day is None:
                # get closest
                idx, val = find_closest_not_null(t, i)
                # print('Closest idx: ', idx)
                if idx > i:
                    # go backward in days
                    r = idx - i
                    new_val = next_backward(val, r)
                else:
                    # go forward in days
                    r = i - idx
                    new_val = next_forward(val, r)

                print(
                    f"Fixing {month_name}, day ",
                    t[i],
                    ", with NONE value, to new val: ",
                    new_val,
                )

                month_data[k][__day_str__] = new_val

    return start_data


def save(data: Dict, dst: str):
    with open(dst, "w") as fp:
        json.dump(data, fp, indent=4)

    print("End of program")


def main():
    p, _ = os.path.splitext(curr_file)
    year = p.split(os.sep)[-1]

    base_dir = os.path.join(os.getcwd(), p)
    destination_dir = os.path.join(base_dir, "original")
    crop_dir = os.path.join(base_dir, "crops")
    words_dir = os.path.join(base_dir, "words")

    logging.info(f"Writing pages in {destination_dir}")

    extract_pages(curr_file, destination_dir)

    extract_crops(destination_dir, crop_dir)

    data = extract_words(crop_dir, visualize=True)

    # with open('result.json', 'r') as fp:
    #     data = json.load(fp)

    data = validate(data)

    save(data, f"result_{year}.json")


if __name__ == "__main__":
    main()
