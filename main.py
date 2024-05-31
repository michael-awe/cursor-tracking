import json
import cv2
import os

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
    fps = video.get(cv2.CAP_PROP_FPS)

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
    try:
    
        # Load the image
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the cursor template
        cursor_template = cv2.imread('cursor.png', 0)

        # Perform template matching
        res = cv2.matchTemplate(gray, cursor_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Get the cursor coordinates
        cursor_x, cursor_y = max_loc

        #Draw a rectangle around the cursor
        cv2.rectangle(image, max_loc, (max_loc[0] + cursor_template.shape[1], max_loc[1] + cursor_template.shape[0]), (0, 0, 255), 2)
        
        #save image to folder
        filename = image_path.split('/')[-1]
        cv2.imwrite(f'processed_frames/{filename}', image)


        return cursor_x, cursor_y
    except Exception as e:
        return None, None
    

    # Display the image with the cursor marked
    # cv2.startWindowThread()
    # cv2.namedWindow("preview")
    # cv2.imshow("preview", image)

    
def main():
    video_path = 'screen_capture.mp4'
    frames = vid_to_frames(video_path)
    process_frames(frames)

if __name__ == '__main__':
    main()