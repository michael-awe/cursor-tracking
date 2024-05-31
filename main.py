import json
import cv2
import os

def load_templates():

    template_paths = os.listdir('templates')
    templates = []
    for path in template_paths:
        template = cv2.imread(f"templates/{path}", 0)
        if template is not None:
            templates.append(template)

    return templates

def vid_to_frames(filename, FPS=30):
    frame_map = {}
    try:
        os.mkdir('frames')
    except:
        #clear the frames folder
        files = os.listdir('frames')
        for file in files:
            os.remove(f'frames/{file}')

    # Open the video file
    video = cv2.VideoCapture(filename)
    # Get the frames per second (fps) of the video
    # fps = video.get(cv2.CAP_PROP_FPS)
    fps = 5


    # Create a counter to name the frames
    frame_count = 0

    # Loop through the video frames
    while True:
        # Read a frame from the video
        success, frame = video.read()

        # Check if the frame was read successfully
        if success:
            # Save the frame as an image file
            cv2.imwrite(f'frames/output_frame_{frame_count}.png', frame)
            frame_count += 1

            timestamp = round(frame_count / fps, 2)

            frame_map[frame_count] = timestamp
            # print(timestamp)
        else:
            # Break the loop if there are no more frames
            break

    # Release the video capture object
    video.release()

    return frame_map

def process_frames(frames):

    processed_frames = []
    try:
        os.mkdir('processed_frames')
    except:
        #clear the frames folder
        files = os.listdir('processed_frames')
        for file in files:
            os.remove(f'processed_frames/{file}')


    for frame in frames:
        timestamp = frames[frame]
        # print(f'Processing frame {frame} at timestamp {timestamp}')

        path = f"frames/output_frame_{frame}.png"

        x,y = find_cursor(path)
        processed_frames.append({
            'frame': frame,
            'timestamp': timestamp,
            'x': x,
            'y': y
        
        })

    #write processed frames to json file
    with open('processed_frames.json', 'w') as f:
        print('WRITING PROCESSED LOG TO processed_frames.json')
        json.dump(processed_frames, f, indent=4)

    print("PROCESSED IMAGES SAVED TO processed_frames FOLDER")

def find_cursor(image_path):
    
    templates = load_templates()
    template_paths = os.listdir('templates')

    best_match_val = -1
    best_match_loc = None
    best_template_path = None

    try:
        for cursor_template, template_path in zip(templates, template_paths):

            # Load the image
            image = cv2.imread(image_path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            res = cv2.matchTemplate(gray, cursor_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Check if the current match is better than the previous best match
            if max_val > best_match_val:
                best_match_val = max_val
                cursor_x, cursor_y = max_loc
                best_path = template_path

            if best_match_val < 0.7:
                print("No cursor found")
                return None, None
            #Draw a rectangle around the cursor
            cv2.rectangle(image, max_loc, (max_loc[0] + cursor_template.shape[1], max_loc[1] + cursor_template.shape[0]), (0, 0, 255), 2)
            
            #save image to folder
            filename = image_path.split('/')[-1]
            cv2.imwrite(f'processed_frames/{filename}', image)
        
    except Exception as e:
        return None, None

    # print(f"Confidence: {best_match_val}")
    # print(f"Template: {best_path}")
    # print(f"Cursor position: {cursor_x}, {cursor_y})")
    
    return None, None

    
def main():
    video_path = 'screen_capture.mp4'
    frames = vid_to_frames(video_path)
    process_frames(frames)

if __name__ == '__main__':
    main()