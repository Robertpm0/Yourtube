import concurrent.futures
import multiprocessing
import streamlit as st
import sqlite3
import requests
import numpy    
import isodate
import re
import kaleido
# import tabulate
import numpy as np
import concurrent.futures
from typing import List,Tuple
import pandas as pd
import ast
import matplotlib.pyplot as plt
import boto3
from textblob import TextBlob
import streamlit_shadcn_ui as ui
import openai
from datetime import datetime
import zipfile
from lxml import etree
import os
import gzip
from st_files_connection import FilesConnection
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from wordcloud import WordCloud
import io

from datetime import date

def get_year_bounds(year):
    first_day = date(year, 1, 1)
    last_day = date(year, 12, 31)
    return first_day, last_day

# Example usage:
# first, last = get_year_bounds(2025)
# set in .sreamlit/secrets.toml
# st.set_page_config(layout="wide")
API_KEY=st.secrets.get("ytv3_key","")  or os.getenv("ytv3_key")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY")
TEMP_ZIP_PATH = 'shared_data/uploaded.zip'  # Streamlit will read this


# SQLite Database Configuration
DB_PATH = "ytAnalysis.db"  #used to store duration for a given videokey
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from PIL import Image
import io
import math

def generate_youtube_report_pdf(output_path, chart_buffers, stats_lines, start_date, end_date,svg_logo_path='yta.svg'):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    # ---- PAGE 1: Cover with Logo + Title ----
    # Convert SVG logo to drawing object
    drawing = svg2rlg(svg_logo_path)
    logo_width = 300
    logo_height = 300
    drawing.scale(logo_width / drawing.width, logo_height / drawing.height)
    renderPDF.draw(drawing, c, (width - logo_width) / 2, height / 2)

    # Title under logo
    c.setFont("Helvetica-Bold", 18)
    title = f"Your YouTube Watch History from {start_date} to {end_date}"
    c.drawCentredString(width / 2, height / 2 - 60, title)
    c.showPage()

    # ---- PAGE 2: Overview ----
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 80, "Overview")

    # Centered list of stats
    c.setFont("Helvetica", 12)
    y_start = height - 140
    for line in stats_lines:
        c.drawCentredString(width / 2, y_start, line)
        y_start -= 20

    # Add small logo at bottom-right
    renderPDF.draw(drawing, c, width - 60, 20)
    c.showPage()

    # ---- REMAINING PAGES: 2 charts per page ----
    chart_per_page = 2
    num_pages = math.ceil(len(chart_buffers) / chart_per_page)

    for i in range(num_pages):
        charts = chart_buffers[i * chart_per_page: (i + 1) * chart_per_page]

        y_positions = [height - 80 - 350 * j for j in range(len(charts))]

        for img_buf, y in zip(charts, y_positions):
            img = Image.open(img_buf)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_width, img_height = img.size
            scale = min((width - 100) / img_width, 300 / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.ANTIALIAS)

            img_stream = io.BytesIO()
            img.save(img_stream, format='PNG')
            img_stream.seek(0)

            x_pos = (width - new_size[0]) / 2
            y_pos = y

            # Border rectangle
            c.setStrokeColor(colors.HexColor("#999999"))
            c.rect(x_pos - 5, y_pos - 5, new_size[0] + 10, new_size[1] + 10, stroke=1, fill=0)

            # Draw chart image
            c.drawImage(ImageReader(img_stream), x_pos, y_pos, width=new_size[0], height=new_size[1])

        # Add small logo to bottom-right
        renderPDF.draw(drawing, c, width - 60, 20)
        c.showPage()

    c.save()




def save_charts_with_stats_to_pdf(image_buffers, stats_text, output_pdf_path):
    """
    Saves stats text on the first page and chart images on following pages into a single PDF.
    - image_buffers: list of BytesIO objects (each contains a PNG image)
    - stats_text: string (can contain newlines) to display on the first page
    - output_pdf_path: string path where the PDF will be saved
    """

    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    page_width, page_height = A4

    # Add Stats Text on First Page
    text_obj = c.beginText(40, page_height - 60)
    text_obj.setFont("Helvetica", 12)
    for line in stats_text.split('\n'):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()

    # Add each chart image on a new page
    for img_buf in image_buffers:
        img = Image.open(img_buf)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_width, img_height = img.size

        # Calculate scaling to fit full page with margins
        margin = 40  # 40 points (~0.5 inch)
        max_width = page_width - 2 * margin
        max_height = page_height - 2 * margin
        scale = min(max_width / img_width, max_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        img = img.resize(new_size, Image.ANTIALIAS)

        img_stream = io.BytesIO()
        img.save(img_stream, format='PNG')
        img_stream.seek(0)

        x_pos = (page_width - new_size[0]) / 2
        y_pos = (page_height - new_size[1]) / 2

        c.drawImage(ImageReader(img_stream), x_pos, y_pos, width=new_size[0], height=new_size[1])
        c.showPage()

    c.save()


# def save_charts_to_pdf(image_buffers, output_pdf_path):
#     """
#     Saves a list of image byte buffers as a single PDF.
#     - image_buffers: list of BytesIO objects (each contains a PNG image)
#     - output_pdf_path: string path where the PDF will be saved
#     """

#     c = canvas.Canvas(output_pdf_path, pagesize=A4)
#     width, height = A4  # Default A4 size in points
#     print(image_buffers)
#     for img_buf in image_buffers:
#         img = Image.open(img_buf)

#         # Convert image to RGB if needed
#         if img.mode != 'RGB':
#             img = img.convert('RGB')

#         # Resize image to fit within A4 page margins (keeping aspect ratio)
#         img_width, img_height = img.size
#         scale = min(width / img_width, height / img_height) * 1.0 # 90% of page size
#         new_size = (int(img_width * scale), int(img_height * scale))
#         img = img.resize(new_size, Image.ANTIALIAS)

#         img_stream = io.BytesIO()
#         img.save(img_stream, format='PNG')
#         img_stream.seek(0)

#         x_pos = (width - new_size[0]) / 2
#         y_pos = (height - new_size[1]) / 2

#         c.drawImage(ImageReader(img_stream), x_pos, y_pos, width=new_size[0], height=new_size[1])
#         c.showPage()  # New page for next image

#     c.save()

# converts seconds integer to days,hours,minutes,seconds and total in hours
def convert_seconds(seconds):
    totHours=seconds //3600
    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return days, hours, minutes, seconds,totHours
# function to get unique video ID given input data

def getUploadID(watchHistory,comments,subs,playLists):
    firstVid=watchHistory['Title'].values[0]
    lastVid=watchHistory['Title'].values[-1]
    firstDate=watchHistory['Date'].values[0]
    lastDate=watchHistory['Date'].values[-1]
    numVids=len(watchHistory)
    numComments=len(comments)
    numSubs=len(subs)
    numLists=len(playLists)
    try:
        parsedFirstVid=firstVid.split(" ")
        firstCharsFirst=[f[0] for f in parsedFirstVid[1:] if f]
    except:
        firstCharsFirst=firstVid
    try:
        parsedLastVid=lastVid.split(" ")
        lastCharsFirst=[f[0] for f in parsedLastVid[1:] if f]
    except:
        lastCharsFirst=lastVid
    
    print(parsedFirstVid)
    print(parsedLastVid)
    id=f"{''.join(firstCharsFirst)}_{''.join(lastCharsFirst)}_{firstDate}_{lastDate}_{numVids}_{numSubs}_{numComments}_{numLists}"
    filename=id
    filename = filename.strip()

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Remove any characters not alphanumeric, dot, underscore, or hyphen
    filename = re.sub(r'[^A-Za-z0-9._-]', '', filename)

    # Optionally, prevent filename from starting with a dot (hidden files)
    if filename.startswith('.'):
        filename = filename[1:]
    return id

def verifyID(ID):
    conn=st.connection('s3',type=FilesConnection)
    allFiles=conn._instance.listdir('thestartofsomethingreat')
    allFiles=[x["Key"] for x in  allFiles]
    print("ALLLLL FILESSSSSS",allFiles)
    substring = ID
    matching_files = [fl for fl in allFiles if substring in fl]
    if len(matching_files)==0:
        return True
    else:
        return False

#takes user data, grabs their ID checks if ID is unique,
# if unique creates in memory CSV's for s3 upload
# returns the in mem s3 uploadable objs
def createFileName(watchHistory,comments,subs,playLists):
    # in memory csv of each dataframe for s3 upload
    dataID=getUploadID(watchHistory,comments,subs,playLists)
    isUnique=verifyID(dataID)
    print("WH",watchHistory)

    if isUnique==False:
        return "False",None,None,None,None
    # whBuffer = io.StringIO()
    # print("WH",watchHistory)
    # watchHistory.to_csv(whBuffer, index=False)
    whBuffer=io.BytesIO()
    print("COmpressing")
    with gzip.GzipFile(fileobj=whBuffer, mode='w') as gz:
        watchHistory.to_csv(gz, index=False, encoding='utf-8')


    cmntsBuffer=io.BytesIO()
    # print("COmpressing")
    with gzip.GzipFile(fileobj=cmntsBuffer, mode='w') as gz3:
        comments.to_csv(gz3, index=False, encoding='utf-8')

    subsBuffer=io.BytesIO()
    # print("COmpressing")
    with gzip.GzipFile(fileobj=subsBuffer, mode='w') as gz2:
        subs.to_csv(gz2, index=False, encoding='utf-8')

    listsBuffer=io.BytesIO()
    # print("COmpressing")
    with gzip.GzipFile(fileobj=listsBuffer, mode='w') as gz4:
        playLists.to_csv(gz4, index=False, encoding='utf-8')
    print("done compressing")
    # whBuffer=watchHistory.to_csv(None).encode()
    # # whBuffer=watchHistory
    # # print(whBuffer)
    # # cmntsBuffer = io.StringIO()

    # # comments.to_csv(cmntsBuffer, index=False)
    # cmntsBuffer=comments.to_csv(None).encode()
    # # subsBuffer = io.StringIO()
    # # subs.to_csv(subsBuffer, index=False)
    # subsBuffer=subs.to_csv(None).encode()
    # # listsBuffer = io.StringIO()
    # listsBuffer=playLists.to_csv(None).encode()
    # # playLists.to_csv(listsBuffer, index=False)

    return dataID,whBuffer,cmntsBuffer,subsBuffer,listsBuffer   


# takes files needing to be uploaded to s3 from user
# check if already uploaded, if not then upload
# boolean funciton returns true if succesful upload 
# else returns false because duplicate data or error
def uploadToS3(files):
    # fileName=''
    s3Path=f'thestartofsomethingreat/'
    for df in files:
        df.replace('', np.nan, inplace=True)

# Drop rows where all values are NaN (were empty or null)
        df = df.dropna(how='all')

    baseID,f1,f2,f3,f4=createFileName(files[0],files[1],files[2],files[3])
    if baseID=="False":
        return False

    watchHistoryPath=f'{baseID}'+'_WH.csv.gz'
    commentsPath=f'{baseID}'+'_CMT.csv'
    subsPath=f'{baseID}'+'_SUB.csv'
    listsPath=f'{baseID}'+'_PLY.csv'
    # print(watchHistoryPath)
    # conn=st.connection('s3',type=FilesConnection)


    s3 = boto3.client(
        service_name="s3",
        region_name="us-east-1",
        aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID", "") or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY", "") or os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    # print(files[0])

    # files[0].to_csv(buff,index=False,encoding='utf-8')
    print("Starting")
    # buff.seek(0)
    f1.seek(0)
    f2.seek(0)
    f3.seek(0)
    f4.seek(0)
    print("STarting")
    s3.upload_fileobj(f1, "thestartofsomethingreat", watchHistoryPath)
    s3.upload_fileobj(f2,"thestartofsomethingreat",commentsPath)
    s3.upload_fileobj(f3,"thestartofsomethingreat",subsPath)
    s3.upload_fileobj(f4,"thestartofsomethingreat",listsPath)
    print("done")
    # print("DONE")
    # with conn.open(watchHistoryPath,'wb') as f:
    #     # print(f1.getvalue())
    #     f.write(f1)
    # print("starting")
    # with conn.open(commentsPath,'wb') as f:
    #     f.write(f2)
    # print("next")
    # with conn.open(subsPath,'wb') as f:
    #     f.write(f3)

    # with conn.open(listsPath,'wb') as f:
    #     f.write(f4)
    return True

def get_video_duration_from_db(video_id: str) -> int:
    """
    Checks if a video ID exists in the SQLite database.
    If found, returns its duration; otherwise, returns -1.

    Args:
        video_id (str): The YouTube video ID to check.

    Returns:
        int: The duration in seconds if found, otherwise -1.
    """
    try:
        conn = sqlite3.connect(DB_PATH,timeout=10,check_same_thread=False)
        cursor = conn.cursor()

        # Query the database for the video ID
        cursor.execute("SELECT duration FROM YTvideo WHERE videoID = ?", (video_id,))
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else -1  # Return duration if found, else -1
    except sqlite3.Error as e:
        print(f"SELECT Database error: {e}")
        conn.close()
        return -1  # Return -1 in case of an error

def save_video_duration_to_db(video_id: str, duration: int):
    """
    Saves a video ID and its duration to the SQLite database.

    Args:
        video_id (str): The YouTube video ID.
        duration (int): Duration in seconds.
    """
    try:
        conn = sqlite3.connect(DB_PATH,timeout=10,check_same_thread=False)
        cursor = conn.cursor()

        # Insert the new record
        cursor.execute("INSERT INTO YTvideo (videoID, duration) VALUES (?, ?)", (video_id, duration))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        conn.close()
        print(video_id)
        print(f"INSERT Database error: {e}")

def get_video_durations(video_ids: List[str]) -> Tuple[dict, int]:
    """
    Fetches the duration of YouTube videos (in seconds) for a list of video IDs.
    If a video ID already exists in the database, it uses the stored duration.

    Args:
        video_ids (List[str]): List of YouTube video IDs.

    Returns:
        Tuple[dict, int]: 
            - A dictionary mapping video IDs to their duration in seconds.
            - The number of failed requests.
    """
    durations = {}
    failed_requests = 0  # Track failures
    video_ids_to_fetch = []  # List of IDs that need API calls

    # Step 1: Check database for existing durations
    for video_id in video_ids:
        durations[video_id]=0
        duration = get_video_duration_from_db(video_id)
        if duration != -1:
            # print("Found Existing Duration")
            durations[video_id] = duration  # Use cached value
        else:
            # print(video_id)
            video_ids_to_fetch.append(video_id)  # Add to API fetch list
    # print(len(video_ids_to_fetch))
    # print(len(set(video_ids)))
    # Step 2: Fetch missing durations from YouTube API
    video_id_chunks = [video_ids_to_fetch[i:i + 50] for i in range(0, len(video_ids_to_fetch), 50)]

    def fetch_videos(chunk):
        """Fetch durations for a chunk of video IDs."""
        nonlocal failed_requests
        params = {
            "part": "contentDetails",
            "id": ",".join(chunk),
            "key": API_KEY
        }
        response = requests.get(YOUTUBE_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("items", []):
                video_id = item["id"]
                duration = isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
                durations[video_id] = int(duration)

                # Store the new duration in the database
                save_video_duration_to_db(video_id, int(duration))
        else:
            failed_requests += 1  # Increment failure count
            save_video_duration_to_db(video_id,0)
            print(f"Failed request: {response.status_code}, {response.text}")
    # print(d)
    # Step 3: Use ThreadPoolExecutor to parallelize API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        executor.map(fetch_videos, video_id_chunks)


    # repo = Repo('.')  # if repo is CWD just do '.'

    # repo.index.add(['YTAnalysis.db'])
    # repo.index.commit('System Database Update')
    # origin = repo.remote('origin')
    # origin.push()
    return durations, failed_requests
 # Print first 10 results

# scrapes html input file with watch history, returns collected data in structured fashion
# pass in html file
def parse_json_with_pd(json_content):

    json_content["isPost"]=json_content["titleUrl"].str.contains("post",case=False)
    json_content["hasChannel"]=json_content["subtitles"].str.contains("name",case=False)
    json_content['subtitles'].fillna(' ',inplace=True)

    json_content["details"].fillna(' ',inplace=True)
    print(len(json_content["details"]))
    print(len(json_content["subtitles"]))
    print(len(json_content['title']))
    ads=[]
    for title in json_content["details"]:
        # print(title[0])
        if title[0]!=' ':
            if 'Ads' in title[0]['name']:
                # titleName=title[0]["name"]
                titleName=True
                ads.append(titleName)


            # print(titleName)
        else:
            titleName=False

            ads.append(titleName)
    json_content["isAd"]=ads




    channels=[]
    x=0
    for title in json_content["subtitles"]:
        # print(title[0])
        if "name" in title[0]:
            titleName=title[0]["name"]
            # print(titleName)
        elif ads[x]==False:
            titleName='deleted/private'
        else:

            titleName='ad'
            
        #     print(title[0])
        #     print(json_content["details"][x][0])
        #     print(json_content["title"][x])
        #     print("_________________________________________")
        # x+=1

        channels.append(titleName)
    

    titles=[]
    for title in json_content["title"]:
        # print(title[0])
        if "Viewed" in title:
            titleName=title.split("Viewed")[1].strip()
            if titleName=='https://www.youtube.com/watch?v=':
                print("y")
                titleName='deleted/private'

        elif "Watched" in title:
            titleName=title.split("Watched")[1].strip()
            if titleName=='https://www.youtube.com/watch?v=':
                print("hi")
                titleName='deleted/private'

        else:
            titleName='deleted/private'



        
        # print(titleName)
        titles.append(titleName)
    json_content["Title"]=titles
    # json_content['Title'] = json_content['title'].apply(lambda s: s.split('Viewed')[1] if 'Viewed' in s else s.split('Watched')[1])
    # print(json_content["Title"])
    json_content["Channel"]=channels
    # json_content['Channel'] = json_content['subtitles'].apply(lambda s: s.split("'name':")[0].split(' ')[0] if 'name' in s else 'ad')
    # print(json_content["Channel"])
    keys=[]
    for key in json_content["titleUrl"]:
        try:
            videoKey = key.split("=")[1]

        except:
            try:
                videoKey=key.split("https://www.youtube.com/post/")[0]

            except:
                print("NO KEY FOUND")
                if key=="https://www.youtube.com/watch?v=":
                    videoKey="Deleted"
                # videoKey='NOKEY'
                else:
                    videoKey=""
        keys.append(videoKey)
    json_content["Key"]=keys

    # multiWatch=json_content.groupby(["titleUrl"]).size().
    # watch_counts = df.groupby(['month','isShort']).size().reset_index(name='count')

    return json_content
        # return videoDates,videoDurations,videoKeys,missedVideos,postsLiked,totalVideosWatched,titles,channels,ads,multiWatch,multiChannel,watchHistory

def parse_html_with_lxml(html_content):
        
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_content, parser)

        main_div = tree.xpath('//div[contains(@class, "mdl-grid")]')[0]  # Find main div
        nested_divs = main_div.xpath('./div')  # Get direct child divs
        totalVideosWatched = len(nested_divs)
        
        # st.text(f"Total videos watched: {totalVideosWatched}")
        
        watchHistory = pd.DataFrame()
        adsBool=[]
        postsBool=[]

        vidTitles=[]
        vidKeys=[]
        channelNames=[]
        watchDates=[]

        titles=[]
        channels=[]
        multiChannel={}
        multiWatch={}
        videoDates, videoDurations, videoKeys = [], [], []
        missedVideos = 0
        ads=0
        postsLiked=0
        for div in nested_divs:
            innerContent = div.xpath('.//div[contains(@class, "mdl-grid")]')
            # if there is np inner content skip we cant collect any data, bad entry
            if not innerContent:
                missedVideos += 1
                continue
            
            dataMembers = innerContent[0].xpath('./div')
            # bad entry, skip
            if len(dataMembers) < 2:
                missedVideos += 1
                continue
            
            videoTitleDiv = dataMembers[1]
            isAdDiv=dataMembers[3]
            isAd=False
            # try:
            #     # print()
            #     if "Details" in isAdDiv.xpath('.//b')[1].text.strip():
            #         # isAd=True
            #         ads+=1
            #         # print("AD WATCHES")
            # except:
            #     pass
            # print(videoTitleDiv.get)
            videoUrlElem = videoTitleDiv.xpath('.//a')
            # no video url, invalid video/entry skip
            if not videoUrlElem:
                missedVideos += 1
                continue
            
            videoUrl = videoUrlElem[0].get("href")
            title=videoUrlElem[0].text.strip()
            # get channel for video
            try:
                channel=videoUrlElem[1].text.strip()

            except:
                # if no channel this is an ad
                isAd=True
                ads+=1
                channel='ad'
            # print(title)
            # if this is the video url the video was deleted 
            if title=='https://www.youtube.com/watch?v=':
                title='Deleted Video/Ad'

            dateSibling = videoTitleDiv.xpath('.//br')[-1]
            date = dateSibling.tail.strip()
            isPost=False
            # if there is a video key it will run this block
            # if not it will be a post not a video
            try:
                videoKey = videoUrl.split("=")[1]
            except:
                postsLiked+=1

                videoKey=videoUrl.split("https://www.youtube.com/post/")[0]
                videoKeys.append(videoKey)
                titles.append(title)
                # print(videoUrl)
                isPost=True
                # print(channel)
                videoDates.append(convert_to_date(date))
                channels.append(channel)
            try:
                if videoKey in videoKeys: # when someone views a vid more than once
                    # tracking number of times a channel was watched and specific videos
                    if videoKey not in multiWatch:
                        # print("key",videoKey)
                        multiWatch[videoKey]=1
                        multiChannel[channel]=1
                    else:
                        multiWatch[videoKey]+=1
                        multiChannel[channel]+=1
                else:
                    prettyDate = convert_to_date(date)
                    videoKeys.append(videoKey)
                    videoDates.append(prettyDate)
                    titles.append(title)
                    channels.append(channel)
            except:
                pass
            # store collected data
            adsBool.append(isAd)
            channelNames.append(channel)
            postsBool.append(isPost)
            vidKeys.append(videoKey)
            watchDates.append(convert_to_date(date))
            vidTitles.append(title)

                
        # compressing collected data to a dict
        watchHistory["isAd"]=adsBool
        watchHistory["isPost"]=postsBool
        watchHistory["Title"]=vidTitles
        watchHistory["Date"]=watchDates
        watchHistory["Key"]=vidKeys 
        watchHistory["Channel"]=channelNames
        return videoDates,videoDurations,videoKeys,missedVideos,postsLiked,totalVideosWatched,titles,channels,ads,multiWatch,multiChannel,watchHistory


def convert_to_date(datetime_str):
        """
        Converts a datetime string in the format "Apr 16, 2024, 10:08:37 AM PST" 
        to just the date "YYYY-MM-DD".

        Args:
        - datetime_str (str): The input datetime string.

        Returns:
        - str: The extracted date in "YYYY-MM-DD" format.
        """
        # Define the format (ignoring the timezone)
        date_obj = datetime.strptime(datetime_str[:-4], "%b %d, %Y, %I:%M:%S %p")
        
        # Convert to date string
        return date_obj.strftime("%Y-%m-%d")



# inits the llm used for parsing watch history dataframe
def createDocumentAgent(df):
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",  # or llama-3, etc.
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    agent = create_pandas_dataframe_agent(llm, df,allow_dangerous_code=True,verbose=False)
    return agent


def plotPlaylists(playlist_names, video_counts, show_in_app=True):
    """
    Plots the number of videos per playlist.
    - playlist_names: list of playlist names
    - video_counts: list of number of videos (same length as playlist_names)
    - show_in_app: if True, displays in Streamlit; else returns PNG bytes
    """

    df = pd.DataFrame({
        'Playlist': playlist_names,
        'Number of Videos': video_counts
    })

    fig = px.bar(
        df,
        x='Playlist',
        y='Number of Videos',
        color_discrete_sequence=['#4DD799'],
        title='' if show_in_app==True else 'Number of Videos per Playlist'
    )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Playlist',
        yaxis_title='Number of Videos',
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

import plotly.express as px
from io import BytesIO
import streamlit as st

def plotWordCloud(df, text_col='clean_text', show_in_app=True):
    """
    Generates and plots a word cloud from a DataFrame column.
    - df: pandas DataFrame
    - text_col: column name containing text data
    - show_in_app: if True, shows in Streamlit; else returns PNG bytes
    """

    all_text = ' '.join(df[text_col].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    plt.title("Comment Cloud")

    if show_in_app:
        st.pyplot(fig)
    else:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
# def plotCommentChart(watch_counts_df, show_in_app=True):
# import plotly.express as px
# from io import BytesIO
# import streamlit as st

def plotCommentChart(watch_counts_df, time_col='month',chartType='comment' ,show_in_app=True):
    """
    Plots total watch counts by month or year as a single-color bar chart.
    - watch_counts_df: DataFrame with columns [time_col, 'count']
    - time_col: 'month' or 'year' (will be used for x-axis)
    - show_in_app: if True, displays in Streamlit; else returns PNG bytes
    """

    if time_col not in watch_counts_df.columns:
        raise ValueError(f"'{time_col}' column not found in DataFrame.")

    fig = px.bar(
        watch_counts_df.sort_values(time_col),
        x=time_col,
        y='count',
        color_discrete_sequence=['#61256F'],
        #title=f'Total Watch Count by {time_col.capitalize()}'
        title='' if show_in_app==True else '# of Comments Left' if chartType=='comment' \
        else 'Average Sentiment of Comments' if chartType=='Sent' else '# Videos Added To Playlist'
    )
    yLab='Comments'
    if chartType=='Sent':
        yLab='Sentiment' 
    elif chartType=='List':
        yLab='Videos'
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title=time_col.capitalize(),
        yaxis_title=yLab,
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

import streamlit as st
import plotly.express as px
from io import BytesIO

def createChannelChart(dataframe, show_in_app=True):
    """
    Generates a horizontal bar chart of watch counts by channel.
    If show_in_app is True, displays it in Streamlit.
    If False, returns PNG bytes for export.
    """

    fig = px.bar(
        dataframe.sort_values('watch_count', ascending=True),
        x='watch_count',
        y='Channel',
        orientation='h',
        color_discrete_sequence=['#77B150'],
        labels={'watch_count': 'Times Watched', 'Channel': 'Channel Name'},
        #title='Channel Watch Count'
        title='' if show_in_app==True else 'Top Channels Watched'
    )

    fig.update_layout(
        xaxis_title='Times Watched',
        yaxis_title='Channel Name',
        plot_bgcolor='white',
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)
                    # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")

                    # st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#E97D62",x_label="Ad Name",y_label="Times Watched")




def createAdChart(dataframe, show_in_app=True):
    """
    Generates a horizontal bar chart of ad watch counts by ad title.
    If show_in_app is True, displays it in Streamlit.
    If False, returns PNG bytes for export.
    """

    fig = px.bar(
        dataframe.sort_values('watch_count', ascending=True),
        x='watch_count',
        y='Title',
        orientation='h',
        color_discrete_sequence=['#E97D62'],
        labels={'watch_count': 'Times Watched', 'Title': 'Ad Name'},
       # title='Ad Watch Count'
       title='' if show_in_app==True else 'Top Ads Watched'
    )

    fig.update_layout(
        xaxis_title='Times Watched',
        yaxis_title='Ad Name',
        plot_bgcolor='white',
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

import plotly.express as px
from io import BytesIO
import streamlit as st
# def createVidPerMonthChart()
    
# def createVidPerMonthChart(df, mode='Month', filter_value=None, show_in_app=True):

def createVidPerMonthChart(watch_counts_df, time_col='month', show_in_app=True):
    """
    Plots a grouped bar chart for watch counts by time column (month or year).
    - watch_counts_df: DataFrame with columns [time_col, 'Keys', 'count']
    - time_col: 'month' or 'year' (used for x-axis)
    - show_in_app: if True, shows in Streamlit; else returns PNG buffer
    """

    fig = px.bar(
        watch_counts_df.sort_values(time_col),
        x=time_col,
        y='count',
        color='Keys',
        barmode='group',
        labels={time_col: time_col.capitalize(), 'count': 'Times Watched', 'Keys': 'Category'},
        # title=f'Watch Counts Grouped by {time_col.capitalize()} and Keys'
        title=''
    )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title=time_col.capitalize(),
        yaxis_title='Times Watched',
        legend_title='Category'
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)
def createVideoChart(dataframe, show_in_app=True):
    """
    Generates a horizontal bar chart of video watch counts by video title.
    If show_in_app is True, displays it in Streamlit.
    If False, returns PNG bytes for export.
    """

    fig = px.bar(
        dataframe.sort_values('watch_count', ascending=True),
        x='watch_count',
        y='Title',
        orientation='h',
        color_discrete_sequence=['#4AB7FF'],
        labels={'watch_count': 'Times Watched', 'Title': 'Video Name'},
        #title='Video Watch Count'
        title='' if show_in_app==True else 'Top Videos Watched'
    )

    fig.update_layout(
        xaxis_title='Times Watched',
        yaxis_title='Video Name',
        plot_bgcolor='white',
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

import plotly.express as px
import streamlit as st
from io import BytesIO

def createVideoChart(dataframe, show_in_app=True):
    """
    Generates a horizontal bar chart of video watch counts.
    - Video titles are not shown on the y-axis.
    - Titles are shown in hover tooltips.
    - Y-axis displays rank numbers to save space.
    """

    # Sort by watch count and create rank column (1 = most watched)
    df = dataframe.copy().sort_values('watch_count', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1  # for y-axis
    df = df.sort_values('watch_count', ascending=True)  # sort for horizontal bars

    fig = px.bar(
        df,
        x='watch_count',
        y='Rank',
        orientation='h',
        color_discrete_sequence=['#4AB7FF'],
        labels={'watch_count': 'Times Watched', 'Rank': 'Video Rank'},
        hover_data={
            'Title': True,         # show full title
            'watch_count': True,   # show watch count
            'Rank': False          # hide rank from hover
        },
        title='' if show_in_app else 'Top Videos Watched'
    )

    fig.update_layout(
        xaxis_title='Times Watched',
        yaxis_title='Video Rank',
        plot_bgcolor='white',
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10))  # smaller font for rank numbers
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

# import plotly.express as px
# from io import BytesIO
# import streamlit as st

def createDurationChart(watch_time_series, show_in_app=True):
    """
    Plots a bar chart of watch time in hours.
    - watch_time_series: pandas Series (index will be used as x-axis)
    - show_in_app: if True, displays in Streamlit; else returns PNG bytes
    """

    df_plot = watch_time_series.copy() / 3600  # Convert to hours
    df_plot = df_plot.reset_index()
    df_plot.columns = ['Index', 'Watch Time (Hours)']

    fig = px.bar(
        df_plot,
        x='Index',
        y='Watch Time (Hours)',
        color_discrete_sequence=['#C054FA'],
        #title='Total Watch Time (Hours)'
        title='' if show_in_app==True else 'Hours of Videos Watched'
    )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='',
        yaxis_title='Watch Time (Hours)',
        showlegend=False
    )

    if show_in_app:
        st.plotly_chart(fig, use_container_width=True)
    else:
        img_bytes = fig.to_image(format='png')
        return BytesIO(img_bytes)

def createImageBuffers(start_date,end_date,view_mode,top_n):
    imStore=[]
    statsToCollect=[]
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

    df["Date"]=pd.to_datetime(df["Date"])
    print(df["Date"])
    df=df[df["isPost"]==False]
    # ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
    statsToCollect.append(f"Videos Watched in Selected Time Period {len(df)} \n")
    

    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    # df.to_csv("savedHistyo.csv",index=False)
    df["Date"]=pd.to_datetime(df["Date"])
    timeWatched=0
    isShort=[]
    for t in df["Duration"]:
        if t!=0 and t<55:
            isShort.append(True)
        else:
            isShort.append(False)
        timeWatched+=t
    d,h,m,s,totHours=convert_seconds(timeWatched)
    # with st.container(border=True):
        # if doExperimental==True:
    statsToCollect.append(f"Total Duration of Videos Watched from {start_date} to {end_date}: {d} Days, {h} Hours, {m} Minutes, {s} Seconds \n")
    
    df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]

    statsToCollect.append(f"Total Comments Left: {len(df)} \n")
    
    def extract_text(entry):
        try:
            data = ast.literal_eval(entry)  # Safely convert string to dict
            return data.get("text", "")  # Remove asterisks
        except:
            return ""

    df['clean_text'] = df['Comment Text'].apply(extract_text)
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity
    df['sentiment'] = df['clean_text'].apply(get_sentiment)
    statsToCollect.append(f'Average Sentiment of your Comments: {df["sentiment"].mean()} \n')
    statsToCollect.append(f'Total Number of Subscriptions: {len(st.session_state["subs"])}')
    statsToCollect.append(f'Total Number of Playlists: {(len(st.session_state["Playlists"]))}')
    statsToCollect.append(f'Total Videos in All Playlists: {len(st.session_state["all_lists"])})')
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    df=df[(df['isAd']==False) & (df["isPost"]==False)]
    video_counts = (
df.groupby(["Key", "Title"])
.size()
.reset_index(name="watch_count")
.sort_values(by="watch_count", ascending=False)
.head(top_n)
)
    # st.subheader(f"Top {top_n} Videos Watched")
    # st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#4AB7FF",y_label="Times Watched",x_label="Video Name")
    topVideos=createVideoChart(video_counts,show_in_app=False)
    imStore.append(topVideos)

    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    df=df[df['isAd']==True]
    # print(df.head())
    # df.to_csv("adscsv.csv",index=False)
    video_counts = (
df.groupby(["Key", "Title"])
.size()
.reset_index(name="watch_count")
.sort_values(by="watch_count", ascending=False)
.head(top_n)
)

    # st.subheader(f"Top {top_n} Ads Watched")
    # # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")

    topAds=createAdChart(video_counts,show_in_app=False)
    imStore.append(topAds)
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    df=df[(df['isAd']==False) & (df["isPost"]==False)]
    video_counts = (
df.groupby(["Channel"])
.size()
.reset_index(name="watch_count")
.sort_values(by="watch_count", ascending=False)
.head(top_n)
)
    topChannels=createChannelChart(video_counts,show_in_app=False)
    imStore.append(topChannels)
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

    df["Date"]=pd.to_datetime(df["Date"])
    print(df["Date"])
    df=df[df["isPost"]==False]
    df["Keys"]=["Is Advertisement" if x==True else "Not Advertisement" for x in df["isAd"].values]
    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
        watch_counts = watch_counts.sort_values('month')
        print(watch_counts)
        # st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()
        vpm=createVidPerMonthChart(watch_counts,time_col='month',show_in_app=False)

    else:
        df['year'] = df['Date'].dt.year
        watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
        # st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')
        vpm=createVidPerMonthChart(watch_counts,time_col='year',show_in_app=False)
    imStore.append(vpm)
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    df["Date"]=pd.to_datetime(df["Date"])
    timeWatched=0
    isShort=[]
    for t in df["Duration"]:
        if t!=0 and t<55:
            isShort.append(True)
        else:
            isShort.append(False)
        timeWatched+=t
    df['isShort']=isShort
    df=df[df['isAd']==False]
    df["Keys"]=["Is a Short" if x==True else "Not a Short" for x in df["isShort"].values]
    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
        watch_counts = watch_counts.sort_values('month')
        print(watch_counts)
        # st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()                        createVidPerMonthChart(watch_counts,time_col='month')
        vpm2=createVidPerMonthChart(watch_counts,time_col='month',show_in_app=False)

    else:
        df['year'] = df['Date'].dt.year
        watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
        # st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')
        vpm2=createVidPerMonthChart(watch_counts,time_col='year',show_in_app=False)
    imStore.append(vpm2)
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

    df["Date"]=pd.to_datetime(df["Date"])
    print(df["Date"])
    df=df[(df['isAd']==False) & (df["isPost"]==False)]


    if view_mode == "Month":
                df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                # watch_counts = watch_counts.sort_values('month')
                watch_time = (
        df.groupby('month')['Duration']
        .sum()
        .rename("watchTotal")
    )
                print(watch_time)
                # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                durs=createDurationChart(watch_time,show_in_app=False)
    else:
                df['year'] = df['Date'].dt.year
                # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                # watch_counts = watch_counts.sort_values('month')
                watch_time = (
        df.groupby('year')['Duration']
        .sum()
        .rename("watchTotal")
    )
                print(watch_counts)
                # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                durs=createDurationChart(watch_time,show_in_app=False)
    imStore.append(durs)
    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    # df.to_csv("savedHistyo.csv",index=False)
    df["Date"]=pd.to_datetime(df["Date"])
    timeWatched=0
    isShort=[]
    for t in df["Duration"]:
        if t!=0 and t<55:
            isShort.append(True)
        else:
            isShort.append(False)
        timeWatched+=t
    df['isShort']=isShort

    df=df[(df['isAd']==False) & (df["isPost"]==False)]
    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
        # watch_counts = watch_counts.sort_values('month')
        watch_time = (
df[df['isShort']==True].groupby('month')['Duration']
.sum()
.rename("watchTotal")
)
        print(watch_time)
        # st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()
        hours=createDurationChart(watch_time,show_in_app=False)

    else:
        df['year'] = df['Date'].dt.year
        # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
        # watch_counts = watch_counts.sort_values('month')
        watch_time = (
df[df['isShort']==True].groupby('year')['Duration']
.sum()
.rename("watchTotal")
)
        print(watch_counts)
        # st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()
        hours=createDurationChart(watch_time,show_in_app=False)
    imStore.append(hours)
    df2 = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
    # df.to_csv("savedHistyo.csv",index=False)
    df2["Date"]=pd.to_datetime(df2["Date"])
    timeWatched=0
    isShort=[]
    for t in df2["Duration"]:
        if t!=0 and t<55:
            isShort.append(True)
        else:
            isShort.append(False)
        timeWatched+=t
    df2['isShort']=isShort

    df2=df2[(df2['isAd']==False) & (df2["isPost"]==False)]
    df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]

    df["Date"]=pd.to_datetime(df["Date"])
    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        watch_counts = df.groupby(['month']).size().reset_index(name='count')
        watch_counts = watch_counts.sort_values('month')
        print(watch_counts)
        # st.bar_chart(watch_counts.set_index('month'),color="#61256F")  # can also use .bar_chart() 
        cmnts=plotCommentChart(watch_counts,time_col='month',show_in_app=False)
    else:
        df['year'] = df['Date'].dt.year
        watch_counts = df.groupby(['year']).size().reset_index(name='count')
        # st.bar_chart(watch_counts.set_index('year'),color="#61256F")
        cmnts=plotCommentChart(watch_counts,time_col='year',show_in_app=False)
    imStore.append(cmnts)
    def extract_text(entry):
        try:
            data = ast.literal_eval(entry)  # Safely convert string to dict
            return data.get("text", "")  # Remove asterisks
        except:
            return ""

    df['clean_text'] = df['Comment Text'].apply(extract_text)
    wc=plotWordCloud(df,show_in_app=False)
    imStore.append(wc)
    def get_sentiment(text):
                    return TextBlob(text).sentiment.polarity  # Range: -1 (negative) to 1 (positive)
    df['sentiment'] = df['clean_text'].apply(get_sentiment)
    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        watch_counts = df.groupby(['month'])['sentiment'].mean().reset_index(name='count')
        watch_counts = watch_counts.sort_values('month')
        # print(watch_counts)
        # st.bar_chart(watch_counts.set_index('month'),color="#C4ED3B")  # can also use .bar_chart() 
        sent=plotCommentChart(watch_counts,time_col='month',chartType='Sent',show_in_app=False)

    else:
        df['year'] = df['Date'].dt.year
        watch_counts = df.groupby(['year'])['sentiment'].mean().reset_index(name='count')
        
        # st.bar_chart(watch_counts.set_index('year'),color="#C4ED3B")
        sent=plotCommentChart(watch_counts,time_col='year',chartType='Sent',show_in_app=False)
    imStore.append(sent)
    lengths=[]
    i=0

    for x in st.session_state['list_names']:
        # if x ==st.session_state['list_names'][0]:
        #     i+=1
        #     continue
        # else:
        lengths.append(len(st.session_state['Playlists'][i]))
        i+=1
    st.session_state['list_names']=[g.replace("Takeout/YouTube and YouTube Music/playlists/","") for g in st.session_state['list_names']]
    # st.bar_chart(pd.DataFrame({"Playlist":st.session_state['list_names'],'Number of Videos':lengths}),x='Playlist',y='Number of Videos',color="#4DD799")
    totlists=plotPlaylists(st.session_state["list_names"],lengths,show_in_app=False)
    imStore.append(totlists)
    df=st.session_state['all_lists']
    # df = st.session_state.history[((pd.to_datetime(stjj.session_state['all_lists']["Playlist Video Creation Timestamp"])).dt.to_period('M').dt.to_timestamp() >= pd.to_datetime(start_date).date()) & (pd.to_datetime(st.session_state['all_lists']["Playlist Video Creation Timestamp"]) <= pd.to_datetime(end_date).date())]

    df["Date"]=pd.to_datetime(df['Playlist Video Creation Timestamp'])
    df["Date2"]=df["Date"].dt.date
    df=df[(df["Date2"]>=pd.to_datetime(start_date).date()) & (df["Date2"]<=pd.to_datetime(end_date).date())]

    if view_mode == "Month":
        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
        watch_counts = df.groupby(['month']).size().reset_index(name='count')
        watch_counts = watch_counts.sort_values('month')
        # print(watch_counts)
        # st.bar_chart(watch_counts.set_index('month'),color="#FF1988")  # can also use .bar_chart() 
        perList=plotCommentChart(watch_counts,time_col='month',chartType='List',show_in_app=False)

    else:
        df['year'] = df['Date'].dt.year
        watch_counts = df.groupby(['year']).size().reset_index(name='count')
        # st.bar_chart(watch_counts.set_index('year'),color="#FF1988")
        perList=plotCommentChart(watch_counts,time_col='year',chartType='List',show_in_app=False)
    imStore.append(perList)
    # save_charts_to_pdf(imStore,"TESTEXPORT.pdf")
    # save_charts_with_stats_to_pdf(imStore,statsToCollect,"STATS.pdf")
    generate_youtube_report_pdf("YTReport.pdf",imStore,statsToCollect,start_date,end_date)

if __name__=="__main__":
    multiprocessing.freeze_support()
    # co
    # initialize globally used datavars
    st.set_page_config(layout='wide')
    if 'collected' not in st.session_state:
        st.session_state["collected"]=False
        st.session_state["vidFrame"]=pd.DataFrame()
        st.session_state["vidFrame2"]=pd.DataFrame()
        st.session_state["f"]=""
        st.session_state["comments"]=""
        st.session_state["rep"]=""
        st.session_state["history"]=""
        st.session_state["durs"]=""
        st.session_state["ads"]=""
        st.session_state["subs"]=""
        st.session_state["Playlists"]=""
        st.session_state["list_names"]=""
        st.session_state["all_lists"]=""



    c1,c2,c3=st.columns(3,gap='small')

    # logo image

    st.image("yta.svg")
    # link to tutorial
    # ui.link_button(text="How to Use", url="https://docs.google.com/document/d/13R3wwBrTg773rhEE1MFx6w3H1lg4gg-GpiKx72tvrLg/edit?usp=sharing", key="link_btn")

    st.markdown('''### Step 1 — Get Your YouTube Data File  
Get your YouTube data file [here](https://takeout.google.com/). Please [Watch our Tutorial](https://youtu.be/PWbsHPSMCKw) first.

---

### Step 2 — Upload the File to This Tool  
Once you've downloaded your data, upload the file using the uploader below to begin interacting with your data.
''')
    # st.link_button("How to Get your Youtube Data File","https://youtu.be/PWbsHPSMCKw",icon='❓')
    st.markdown("---")
    # st.info("This tool DOES NOT collect your data. Feel free to review our open source codebase to verify our claims.")
    # input data zip file
    c1,c2,c3=st.columns(3,gap='large')
    uploaded_file = st.file_uploader("Upload File From Google Takeout", type=["zip"])

    st.markdown("",help="By uploading, you are agreeing to the use of your data with our services")

    # uploaded_file=TEMP_ZIP_PATH = 'shared_data/uploaded.zip'  # Streamlit will read this

    # doExperimental=ui.checkbox(label="")
    doExperimental=True

    # st.session_state["f"]=uploaded_file
    # doExperimental=st.checkbox("Get Watchtime (Experimental)")
    # st.session_state.sp=st.progress()
    if st.button("Visualize",icon='👀'):
        st.session_state["collected"]=False
        st.session_state["SavedPDF"]=False
        st.session_state["vidFrame"]=pd.DataFrame()
        st.session_state["vidFrame2"]=pd.DataFrame()
        st.session_state["f"]=""
        st.session_state["comments"]=""
        st.session_state["rep"]=""
        st.session_state["history"]=""
        st.session_state["durs"]=""
        st.session_state["ads"]=""
        st.session_state["subs"]=""
        st.session_state["Playlists"]=""
        st.session_state["list_names"]=""
        st.session_state["all_lists"]=""
        st.toast("Analyzing Watch History")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Read the uploaded file into a BytesIO buffer
        # with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        st.session_state["sp"]=st.progress(0,"Starting up Processing")
        # st.session_state.sp=st.progress(0,"Starting up Processing")
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        # with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:

            # List all files in the ZIP
            st.session_state.sp.progress(10,"Reading Data Files")
            file_list = zip_ref.namelist()
            #print(file_list)
            # Specify the folder and the HTML file to extract
            target_folder = "Takeout/YouTube and YouTube Music/history/"
            target_html_file = "watch-history.html"  # Change as needed
            target_html_file="watch-history.json"
            target_folder2 = "Takeout/YouTube and YouTube Music/comments/"
            target_folder3 = "Takeout/YouTube and YouTube Music/subscriptions/"


            target_html_file2='comments.csv'
            target_html_file3='subscriptions.csv'

            
            # Construct the full path of the target HTML file
            full_path = f"{target_folder}{target_html_file}"
            full_path2=f"{target_folder2}{target_html_file2}"
            full_path3=f"{target_folder3}{target_html_file3}"
            commentsFiles=[f for f in zip_ref.namelist() if 'comments' in f and '.csv' in f]
            commentsdfs=[]
            if full_path2 in file_list:
                for commentdf in commentsFiles:
                    with zip_ref.open(commentdf) as file:
                        df=pd.read_csv(file)
                        commentsdfs.append(df)
                
                st.session_state["comments"]=pd.concat(commentsdfs,ignore_index=True)
                st.session_state["comments"]["Date"]=pd.to_datetime(st.session_state["comments"]["Comment Create Timestamp"])

                # with zip_ref.open(full_path2,'r') as csv_file:
                #     commentsDf=pd.read_csv(csv_file)
                #     commentsDf["Date"]=pd.to_datetime(commentsDf["Comment Create Timestamp"])
                #     st.session_state["comments"]=commentsDf

            # full_path = f"{target_folder}{target_html_file}"
            # full_path2=f"{target_folder2}{target_html_file2}"
            if full_path3 in file_list:
                with zip_ref.open(full_path3,'r') as csv_file:
                    subsDf=pd.read_csv(csv_file)
                    # subsDf["Date"]=pd.to_datetime(subsDf["Comment Create Timestamp"])
                    st.session_state["subs"]=subsDf


          
            # List of files in the folder, filtering CSVs that aren't the excluded file
            csv_files = [f for f in zip_ref.namelist() if 'playlists' in f and '.csv' in f and 'playlists.csv' not in f]
            # print(csv_files)
            # Read and parse each CSV file
            dataframes = []
            for csv_file in csv_files:
                with zip_ref.open(csv_file) as file:
                    df = pd.read_csv(file)
                    dataframes.append(df)
            st.session_state["Playlists"]=dataframes
            st.session_state["list_names"]=[f.replace(".csv",'') for f in csv_files]
        # Now `dataframes` is a list of DataFrames from all parsed CSVs
        # Optionally combine them into a single DataFrame
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                good=True
            except:
                # st.rerun()
                st.error("Error!: Incorrect file uploaded to tool. See the 'How To Use' Button above for what your file should contain! ")
                # st.rerun()
                good=False
            # if combined_df:
            if good==False:
                st.stop()


            st.session_state['all_lists']=combined_df

        #     # Example: print the combined DataFrame
        #     print(combined_df)
            #print(full_path)
            if full_path in file_list:

                # print("yo")
                # Extract and read the HTML file
                with zip_ref.open(full_path,'r') as html_file:
                    df=pd.read_json(html_file)
                    # print(df.head())
                    # df.to_csv("json22csv1.csv",index=False)
                    # print("dec")
                    # # Normalize the JSON structure
                    # df = pd.json_normalize(html_file,
                    #    meta=['header', 'title', 'titleUrl', 'time', 'products', 'activityControls'],
                    #    errors='ignore')
                    # df.to_csv("josncsv.csv",index=False)

# # Handle entries that do not have "subtitles"
#                     df_no_subtitles = pd.json_normalize(
#     [entry for entry in html_file if 'subtitles' not in entry],
#     sep='_'
# )

# # Combine both into one DataFrame
#                     final_df = pd.concat([df, df_no_subtitles], ignore_index=True, sort=False)
#                     final_df.to_csv("json2csv2.csv",index=False)
# # Display
# print(final_df.head())
                    # html_content = html_file.read().decode("utf-8")

                # df=pd.read_html(html_content)

                # Parse with BeautifulSoup
                # print("got")
                # print(len(df))
                    # df=pd.read_json()
                
                parsedJson=parse_json_with_pd(df)  
                st.session_state.sp.progress(45,"Preparing  Watch History")
                # videoDates,videoDurations,videoKeys,missedVideos,postsLiked,tot,titles,channels,numAds,repeats,repeatChannel,history=parse_html_with_lxml(html_content)
                # st.session_state["vidFrame"]["WatchDate"]=videoDates
                # st.session_state["vidFrame"]["VideoKey"]=videoKeys
                # st.session_state["vidFrame"]["VideoTitle"]=titles
                # st.session_state.f=repeatChannel
                # st.session_state.reps=repeats
                # st.session_state["vidFrame"]["Channel"]=channels


                # soup = BeautifulSoup(html_content, "html.parser")
                # # print("jj")
                # main_div = soup.find("div", {"class": "mdl-grid"})  # Find div with id="main"
                # # print("prepping")
                # # Find all divs within the main div
                # nested_divs = main_div.find_all("div",recursive=False)
                # totalVideosWatched=len(nested_divs)
                # st.text(fr"toal vids watched: {totalVideosWatched}")
                # watchHistory=pd.DataFrame()
                # # Print all nested divs
                # videoDates=[]
                # videoDurations=[]
                # videoKeys=[]
                # missedVideos=0
                # for div in nested_divs:
                #     innerContent=div.find("div",{"class":"mdl-grid"})
                #     dataMembers=innerContent.find_all("div",recursive=False)
                #     videoTitleDiv=dataMembers[1]
                #     videoUrl=videoTitleDiv.find_all("a",recursive=False)[0].get("href")
                #     videoKey=videoUrl.split("=")[1]
                #     dateSibling=videoTitleDiv.find_all("br",recusive=False)[-1]
                #     date=dateSibling.next_sibling.strip()
                #     prettyDate=convert_to_date(date)
                #     # print(prettyDate)
                #     # print(videoKey)

                #     # print(getYTVideoDuration("VeNfHj6MhgA"))
                #     #videoDuration=getYTVideoDuration(videoKey)
                #     # if videoDuration==0:
                #     #     missedVideos+=1
                #     videoKeys.append(videoKey)
                #     videoDates.append(prettyDate)
                #    # videoDurations.append(videoDuration)
                #     # st.text(rf"{videoUrl}")
                #     # print("\nNested Div:", div)
                #     # print("Div Class:", div.get("class"))
                #

                st.toast("Starting Video Duration Collection") 

                #inc=100/tot
                # import numpy as np
                # inc = np.linspace(0,1,tot)
                #inc =list(range(0,tot))
                #st.text(fr"{len(inc)}")
                # global mBar
                # print(inc)
                history=parsedJson
# Example usage:
                # video_ids = ["VIDEO_ID_1", "VIDEO_ID_2", ..., "VIDEO_ID_30000"]  # Replace with actual video IDs
                if doExperimental==True:
                    st.session_state.sp.progress(75,"Getting Video Durations (this may take several minutes)")
                    videoDurations,mv = get_video_durations(history["Key"])
                else:
                    videoDurations=[0]*len(history)
                # print(len(videoKeys))
                # print(len((videoKeys))))
                # Print a sample result
                # print(list(videoDurations.keys())[:10]) 
                # print(list(videoKeys)[:10]) 
                # print(videoDurations)
                # st.session_state["vidFrame"]["Duration"]=list(videoDurations.values())
                st.session_state["durs"]=videoDurations
                st.session_state.sp.progress(90,"Generating Charts and Stats")

                # with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                #     #mBar=st.progress(0.0,"")
                #     # st.session_state["progressBar"]=st.progress(0.0,"Video Duration Collection Process!")
                #     videoDurations,badVids = list(zip(executor.map(getYTVideoDuration, videoKeys,inc,chunksize=1000)))
                # #print()
                # st.session_state["vidFrame2"]=st.session_state["vidFrame"].copy()\

                if doExperimental==True:
                    history['Duration'] = history['Key'].map(videoDurations).fillna(0).astype(int)
                else:
                    history["durs"]=[0]*len(history)
                    history['Duration']=history["durs"]
                history['Date']=pd.to_datetime(history["time"],format='ISO8601').dt.date
                # print(history['Date'])
                h1Count=len(history)
                # history.to_csv("yo.csv",index=False)
                s3Files=[history,st.session_state["comments"],subsDf,combined_df]
                # uploaded=uploadToS3(s3Files)

                # if uploaded==True:
                #     st.balloons()
                # else:
                #     st.snow()

                history=history[history["Duration"]<90000]

                h2Count=len(history)
                # history.to_csv("yo.csv",index=False)
                st.session_state.history=history
                st.session_state.collected=True
                st.session_state.sp.empty()

            else:
                st.error("No JSON found in zip file, pelase ensure json format is selected for watch history in the Google Takeout export options.")
    if st.session_state.collected==True:

        tabControl=ui.tabs(["Overview","Yearly Snapshot","Watch History","Comments","Subsctripions & Playlists","AI"],default_value="Overview")
        st.sidebar.subheader("Adjustments")
        view_mode = st.sidebar.radio("Group videos watched by:", ["Month", "Year"])

        top_n = st.sidebar.slider("Select number of top values to display:", min_value=1, max_value=20, value=10)
        
        min_date=st.session_state.history["Date"].min()
        max_date=st.session_state.history["Date"].max()
        # start_date=True
        # end_date=True
    #     try:
    #         start_date, end_date = st.sidebar.date_input(

    #     "Select date range:",
    #     [min_date, max_date],
    #     min_value=min_date,
    #     max_value=max_date
    # )
    #     except:
    #         start_date,end_date=None,None
    #         st.sidebar.text("Date Range Incomplete!")

        # st.session_state["vidFrame"]["WatchDate"]=pd.to_datetime(st.session_state["vidFrame"]["WatchDate"]).dt.date
        # st.session_state.history["Date"]=pd.to_datetime(history["Date"]).dt.date
        # st.session_state.sp.progress(100,"Creating UI")
        st.toast("Loading UI")
        # firstRow=st.columns(2)
        start_date,end_date=min_date,max_date
        if start_date and end_date:
            if tabControl in ["Watch History","Comments","Subsctripions & Playlists","AI"]:
                try:
                    start_date, end_date = st.sidebar.date_input(

        "Select date range:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
                except:
                    start_date,end_date=None,None
                    st.sidebar.text("Date Range Incomplete!")
            else:
                st.sidebar

            if st.session_state["SavedPDF"]==False:
                # createImageBuffers(start_date,end_date,view_mode,top_n)
                st.session_state["SavedPDF"]=True
            if tabControl=='Overview':
                st.subheader("Overview")
                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                df["Date"]=pd.to_datetime(df["Date"])
                timeWatched=0
                isShort=[]
                for t in df["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                d,h,m,s,totHours=convert_seconds(timeWatched)
                st.text(f"You have Watched {d} days, {h} hours, {m} minutes, and {s} seconds worth of videos.")
                sections=st.columns(3)

                with sections[0]:

                # with firstRow[0]:
                    with st.container(border=True):
                        st.subheader("Stats")
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                        df["Date"]=pd.to_datetime(df["Date"])
                        print(df["Date"])
                        df=df[df["isPost"]==False]
                        ui.metric_card("Total Number of Videos Watched",f"{len(df)}")
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                        df["Date"]=pd.to_datetime(df["Date"])
                        timeWatched=0
                        isShort=[]
                        for t in df["Duration"]:
                            if t!=0 and t<55:
                                isShort.append(True)
                            else:
                                isShort.append(False)
                            timeWatched+=t
                        df['isShort']=isShort
                        df=df[df['isAd']==False]
                        df=df[df['isShort']==True]
                        df=df[df["isPost"]==False]

                        ui.metric_card("Number of Shorts Watched",len(df))
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                        df=df[df['isAd']==True]
                        ui.metric_card("Number of Ads Watched",f"{len(df)}")
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                        df=df[df['isPost']==True]
                        ui.metric_card("Number of Posts Seen",f'{len(df)}')
                        df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                        def extract_text(entry):
                            try:
                                data = ast.literal_eval(entry)  # Safely convert string to dict
                                return data.get("text", "")  # Remove asterisks
                            except:
                                return ""
                        def get_sentiment(text):
                            return TextBlob(text).sentiment.polarity
                        
                        df['clean_text'] = df['Comment Text'].apply(extract_text)
                        df['sentiment'] = df['clean_text'].apply(get_sentiment)
                        av=df['sentiment'].mean()
                        if av<0:
                            cmnSent='Mostly Angry'
                        if av>0:
                            cmnSent='Mostly Happy'
                        else:
                            cmnSent='Sometimes happy, sometimes mad'
                        ui.metric_card(f"Number of Comments Left",f"{len(df)}")
                        ui.metric_card("Your Comments Are",cmnSent)
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                        df["Date"]=pd.to_datetime(df["Date"])
                        print(df["Date"])
                        df=df[df["isPost"]==False]
                        daysDelta=(end_date-start_date).days
                        ui.metric_card("Videos per Day",f"{int(len(df)/daysDelta)}")
                        # st.text('or')
                        ui.metric_card("Hours per Day",f"{round(totHours/daysDelta,2)}")
                        ui.metric_card("Fist Watched",f"{df['Title'].values[-1]}")
                        ui.metric_card("Last Watched",f"{df['Title'].values[0]}")

                with sections[1]:

                # with firstRow[0]:
                    with st.container(border=True):

        
            # top videos watched
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                        df=df[(df['isAd']==False) & (df["isPost"]==False)]
                        video_counts = (
                    df.groupby(["Key", "Title"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )
                        st.subheader(f"Top {top_n} Videos Watched")
                        # st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#4AB7FF",y_label="Times Watched",x_label="Video Name")
                        createVideoChart(video_counts)


            # top ads watched 
                    # with firstRow[1]:
                    with st.container(border=True):
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                        df=df[df['isAd']==True]
                        print("AD LEN",len(df[df['Key']==rf'https://www.youtube.com/watch?v=']))
                        # print(df.head())
                        # df.to_csv("adscsv.csv",index=False)
                        print(df['Title'])
                        video_counts = (
                    df.groupby(["Key", "Title"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )

                        st.subheader(f"Top {top_n} Ads Watched")
                        # # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")
                        # print(video_counts[video_counts['Title']=='https://www.youtube.com/watch?v='])
                        print("______________")
                        print(video_counts)
                        for x in video_counts['Title'].values:
                            print(x.split('https://www.youtube.com/watch?v='))
                        createAdChart(video_counts)

                # top channels watched

                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                    df=df[(df['isAd']==False) & (df["isPost"]==False)]
                    
                    video_counts = (
                    df.groupby(["Channel"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )
                    with st.container(border=True):

                        st.subheader(f"Top {top_n} Channels Watched")
                        # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")
                        createChannelChart(video_counts)


                # ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
                with sections[2]:
                    with st.container(border=True):
                        df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]

                        st.subheader(f"Comments left per Year")
                    # df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                        df["Date"]=pd.to_datetime(df["Date"])
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#61256F")
                        plotCommentChart(watch_counts,time_col='year')
                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                    df["Date"]=pd.to_datetime(df["Date"])
                    print(df["Date"])
                    df=df[df["isPost"]==False]              
                # with firstRow[0]:
                    with st.container(border=True):

                        # df["Date"]=pd.to_datetime(df["Date"])
                        print(df["Date"])
                        df=df[(df['isAd']==False) & (df["isPost"]==False)]
                        if doExperimental==True:
                        # --- Group and Aggregate ---
                            st.subheader(f"Total Hours Watched per Year (Shorts + Regular)")


                            df['year'] = df['Date'].dt.year
                            # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df.groupby('year')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            # print(watch_counts)
                            # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                            createDurationChart(watch_time)
                    with st.container(border=True):
                        st.subheader("Videos Added to Playlist per Year")
                        df=st.session_state['all_lists']
                        # df = st.session_state.history[((pd.to_datetime(stjj.session_state['all_lists']["Playlist Video Creation Timestamp"])).dt.to_period('M').dt.to_timestamp() >= pd.to_datetime(start_date).date()) & (pd.to_datetime(st.session_state['all_lists']["Playlist Video Creation Timestamp"]) <= pd.to_datetime(end_date).date())]

                        df["Date"]=pd.to_datetime(df['Playlist Video Creation Timestamp'])
                        df["Date2"]=df["Date"].dt.date
                        df=df[(df["Date2"]>=pd.to_datetime(start_date).date()) & (df["Date2"]<=pd.to_datetime(end_date).date())]
                        # print(df["month"])
                                
                        
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#FF1988")
                        plotCommentChart(watch_counts,time_col='year',chartType='List')
            elif tabControl=='Yearly Snapshot':
                st.header("Year in Review")
                # df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df=st.session_state.history.copy()
                # df=st.session_state.history
                # df["Date"]=pd.to_datetime(df["Date"])
                # # st.head("HI")
                choice=st.selectbox("Choose Year",options=pd.unique(pd.to_datetime(st.session_state.history["Date"]).dt.year))
                p2Cols=st.columns(3)
                startDate,endDate=get_year_bounds(choice)
                with p2Cols[0]:
                    st.subheader(f"Stats for {choice}")
                    with st.container(border=True):
                        
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(startDate).date()) & (st.session_state.history["Date"] <= pd.to_datetime(endDate).date())]
                        df3=df[df['isPost']==False]
                        ui.metric_card(f"Total Videos Watched",f"{len(df3)}")
                        isShort=[]
                        timeWatched=0
                        for t in df["Duration"]:
                            if t!=0 and t<55:
                                isShort.append(True)
                            else:
                                isShort.append(False)
                            timeWatched+=t
                        df['isShort']=isShort
                        df2=df[(df['isShort']==True )& (df['isPost']==False) & (df['isAd']==False)]
                        ui.metric_card(f"Shorts Watched",f"{len(df2)}")
                        df1=df[df['isPost']==True]
                        ui.metric_card("Posts Seen",f"{len(df1)}")
                        df1=df[df['isAd']==True]
                        ui.metric_card("Ads Watched/Seen",f"{len(df1)}")
                        d,h,m,s,totHours=convert_seconds(timeWatched)
                        ui.metric_card("Total Hours of Videos Watched",f"{totHours}")
                        ui.metric_card("Hours Watched Per Day",f"{round(totHours/365,2)}")
                        ui.metric_card("Videos Watched Per Day",f"{round(len(df3)/365,2)}")
                        df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(startDate).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(endDate).date())]

                        ui.metric_card("Total Comments Left",f"{len(df)}")
                with p2Cols[1]:

                    with st.container(border=True):
            # top videos watched
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(startDate).date()) & (st.session_state.history["Date"] <= pd.to_datetime(endDate).date())]
                        df=df[(df['isAd']==False) & (df["isPost"]==False)]
                        video_counts = (
                    df.groupby(["Key", "Title"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )
                        st.subheader(f"Top {top_n} Videos Watched")
                        # st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#4AB7FF",y_label="Times Watched",x_label="Video Name")
                        createVideoChart(video_counts)
            # top ads watched 
                    # with firstRow[1]:
                    with st.container(border=True):
                        df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(startDate).date()) & (st.session_state.history["Date"] <= pd.to_datetime(endDate).date())]
                        df=df[df['isAd']==True]
                        print("AD LEN",len(df[df['Key']==rf'https://www.youtube.com/watch?v=']))
                        # print(df.head())
                        # df.to_csv("adscsv.csv",index=False)
                        print(df['Title'])
                        video_counts = (
                    df.groupby(["Key", "Title"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )
                        st.subheader(f"Top {top_n} Ads Watched")
                        # # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")
                        # print(video_counts[video_counts['Title']=='https://www.youtube.com/watch?v='])
                        print("______________")
                        print(video_counts)
                        for x in video_counts['Title'].values:
                            print(x.split('https://www.youtube.com/watch?v='))
                        createAdChart(video_counts)
                # top channels watched
                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(startDate).date()) & (st.session_state.history["Date"] <= pd.to_datetime(endDate).date())]
                    df=df[(df['isAd']==False) & (df["isPost"]==False)]
                    
                    video_counts = (
                    df.groupby(["Channel"])
                    .size()
                    .reset_index(name="watch_count")
                    .sort_values(by="watch_count", ascending=False)
                    .head(top_n)
                )
                    with st.container(border=True):

                        st.subheader(f"Top {top_n} Channels Watched")
                        # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")
                        createChannelChart(video_counts)


                # ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
                with p2Cols[2]:
                    with st.container(border=True):
                        df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(startDate).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(endDate).date())]

                        st.subheader(f"Comments left per Month")
                    # df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                        df["Date"]=pd.to_datetime(df["Date"])
                        df['month'] = df['Date'].dt.month
                        watch_counts = df.groupby(['month']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#61256F")
                        plotCommentChart(watch_counts,time_col='month')
                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(startDate).date()) & (st.session_state.history["Date"] <= pd.to_datetime(endDate).date())]

                    df["Date"]=pd.to_datetime(df["Date"])
                    # print(df["Date"]) 
                    df=df[df["isPost"]==False]              
                # with firstRow[0]:
                    with st.container(border=True):

                        # df["Date"]=pd.to_datetime(df["Date"])
                        print(df["Date"])
                        df=df[(df['isAd']==False) & (df["isPost"]==False)]
                        if doExperimental==True:
                        # --- Group and Aggregate ---
                            st.subheader(f"Total Hours Watched per Month (Shorts + Regular)")


                            df['month'] = df['Date'].dt.month
                            # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df.groupby('month')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            # print(watch_counts)
                            # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                            createDurationChart(watch_time)
                    with st.container(border=True):
                        st.subheader("Videos Added to Playlist per Year")
                        df=st.session_state['all_lists']
                        # df = st.session_state.history[((pd.to_datetime(stjj.session_state['all_lists']["Playlist Video Creation Timestamp"])).dt.to_period('M').dt.to_timestamp() >= pd.to_datetime(start_date).date()) & (pd.to_datetime(st.session_state['all_lists']["Playlist Video Creation Timestamp"]) <= pd.to_datetime(end_date).date())]

                        df["Date"]=pd.to_datetime(df['Playlist Video Creation Timestamp'])
                        df["Date2"]=df["Date"].dt.date
                        df=df[(df["Date2"]>=pd.to_datetime(startDate).date()) & (df["Date2"]<=pd.to_datetime(endDate).date())]
                        # print(df["month"])
                                
                        
                        df['month'] = df['Date'].dt.month
                        watch_counts = df.groupby(['month']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#FF1988")
                        plotCommentChart(watch_counts,time_col='month',chartType='List')
            # if tabControl=="Watch History":
            #     st. markdown("---")
            if tabControl=="Watch History":
                st. markdown("---")

                st.header("Watch History")
                sections=st.columns(3)
                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                df["Date"]=pd.to_datetime(df["Date"])
                timeWatched=0
                isShort=[]
                for t in df["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                d,h,m,s,totHours=convert_seconds(timeWatched)
                st.text(f"You have Watched {d} days {h} hours {m} minutes {s} seconds worth of videos.")
                # with sections[0]:

                # with firstRow[0]:
                with st.container(border=True):



            # top videos watched
                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                    df=df[(df['isAd']==False) & (df["isPost"]==False)]
                    video_counts = (
                df.groupby(["Key", "Title"])
                .size()
                .reset_index(name="watch_count")
                .sort_values(by="watch_count", ascending=False)
                .head(top_n)
            )
                    st.subheader(f"Top {top_n} Videos Watched")
                    # st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#4AB7FF",y_label="Times Watched",x_label="Video Name")
                    createVideoChart(video_counts)


        # top ads watched 
                # with firstRow[1]:
                with st.container(border=True):
                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                    df=df[df['isAd']==True]
                    # print(df.head())
                    # df.to_csv("adscsv.csv",index=False)
                    video_counts = (
                df.groupby(["Key", "Title"])
                .size()
                .reset_index(name="watch_count")
                .sort_values(by="watch_count", ascending=False)
                .head(top_n)
            )

                    st.subheader(f"Top {top_n} Ads Watched")
                    # # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")

                    createAdChart(video_counts)
            # top channels watched

                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                    df=df[(df['isAd']==False) & (df["isPost"]==False)]
                    video_counts = (
                df.groupby(["Channel"])
                .size()
                .reset_index(name="watch_count")
                .sort_values(by="watch_count", ascending=False)
                .head(top_n)
            )
                with st.container(border=True):

                    st.subheader(f"Top {top_n} Channels Watched")
                    # st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")
                    createChannelChart(video_counts)

                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                df["Date"]=pd.to_datetime(df["Date"])
                print(df["Date"])
                df=df[df["isPost"]==False]
                ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
                # --- Group and Aggregate ---
                with st.container(border=True):

                    st.subheader(f"Videos Watched per {view_mode} (Videos vs Ads)")
                    df["Keys"]=["Is Advertisement" if x==True else "Not Advertisement" for x in df["isAd"].values]
                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
                        watch_counts = watch_counts.sort_values('month')
                        print(watch_counts)
                        # st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()
                        createVidPerMonthChart(watch_counts,time_col='month')
                    else:
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')
                        createVidPerMonthChart(watch_counts,time_col='year')

                    # df=df[df["isPost"]==False]
                    # ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
                    # ui.metric_card()




                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                with st.container(border=True):

                    df["Date"]=pd.to_datetime(df["Date"])
                    timeWatched=0
                    isShort=[]
                    for t in df["Duration"]:
                        if t!=0 and t<55:
                            isShort.append(True)
                        else:
                            isShort.append(False)
                        timeWatched+=t
                    df['isShort']=isShort
                    df=df[df['isAd']==False]
                    df["Keys"]=["Is a Short" if x==True else "Not a Short" for x in df["isShort"].values]

                    st.subheader(f"Videos Watched per {view_mode} (Normal Vs Shorts)")
                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
                        watch_counts = watch_counts.sort_values('month')
                        print(watch_counts)
                        # st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()                        createVidPerMonthChart(watch_counts,time_col='month')
                        createVidPerMonthChart(watch_counts,time_col='month')
 
                    else:
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')
                        createVidPerMonthChart(watch_counts,time_col='year')


                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                df["Date"]=pd.to_datetime(df["Date"])
                timeWatched=0
                isShort=[]
                for t in df["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                d,h,m,s,totHours=convert_seconds(timeWatched)
                with st.container(border=True):
                    if doExperimental==True:
                        st.subheader(f"Total Duration of Videos Watched from {start_date} to {end_date}")
                        c1,c2,c3,c4=st.columns(4)
                        with c1:
                            # st.metric("Days",d)
                            ui.metric_card("Days",d)
                        with c2:
                            # st.metric("Hours",h)
                            ui.metric_card("Hours",h)

                        with c3:
                            # st.metric("Minutes",m)
                            ui.metric_card("Minutes",m)

                        with c4:
                            # st.metric("Seconds",s)
                            ui.metric_card("Seconds",s)


                
                with st.container(border=True):

                    df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                    df["Date"]=pd.to_datetime(df["Date"])
                    print(df["Date"])
                    df=df[(df['isAd']==False) & (df["isPost"]==False)]
                    if doExperimental==True:
                    # --- Group and Aggregate ---
                        st.subheader(f"Total Hours Watched per {view_mode} (Shorts + Regular)")

                        if view_mode == "Month":
                            df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df.groupby('month')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            print(watch_time)
                            # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                            createDurationChart(watch_time)
                        else:
                            df['year'] = df['Date'].dt.year
                            # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df.groupby('year')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            print(watch_counts)
                            # st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                            createDurationChart(watch_time)


                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                df["Date"]=pd.to_datetime(df["Date"])
                timeWatched=0
                isShort=[]
                for t in df["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                df['isShort']=isShort

                df=df[(df['isAd']==False) & (df["isPost"]==False)]

                # --- Group and Aggregate ---
                if doExperimental==True:
                    with st.container(border=True):

                        st.subheader(f"Hours of Short Videos Watched per {view_mode}")

                        if view_mode == "Month":
                            df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df[df['isShort']==True].groupby('month')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            print(watch_time)
                            # st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()
                            createDurationChart(watch_time)

                        else:
                            df['year'] = df['Date'].dt.year
                            # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                            # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                            # watch_counts = watch_counts.sort_values('month')
                            watch_time = (
                    df[df['isShort']==True].groupby('year')['Duration']
                    .sum()
                    .rename("watchTotal")
                )
                            print(watch_counts)
                            # st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()
                            createDurationChart(watch_time)

                # df2=df.copy()
            elif tabControl=="Comments":

                st.markdown('---')
                st.header("Comments")
                df2 = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                # df.to_csv("savedHistyo.csv",index=False)
                df2["Date"]=pd.to_datetime(df2["Date"])
                timeWatched=0
                isShort=[]
                for t in df2["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                df2['isShort']=isShort

                df2=df2[(df2['isAd']==False) & (df2["isPost"]==False)]
                df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                compCols=st.columns(2)
                rate=(len(df)/len(df2))*100
                per=100
                if rate <1:
                    rate = (len(df)/len(df2))*1000
                    per=1000
                if rate <1:
                    rate = (len(df)/len(df2))*10000
                    per=10000
                with st.container(border=True):

                    with compCols[0]:
                        ui.metric_card("Total Comments Left",f"{len(df)}")
                    with compCols[1]:
                        ui.metric_card("Comment Frequency",f"{int(rate)} comments per {per} videos")
                with st.container(border=True):

                    st.subheader(f"Comments left per {view_mode}")
                    # df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                    df["Date"]=pd.to_datetime(df["Date"])
                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        watch_counts = df.groupby(['month']).size().reset_index(name='count')
                        watch_counts = watch_counts.sort_values('month')
                        print(watch_counts)
                        # st.bar_chart(watch_counts.set_index('month'),color="#61256F")  # can also use .bar_chart() 
                        plotCommentChart(watch_counts,time_col='month')
                    else:
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#61256F")
                        plotCommentChart(watch_counts,time_col='year')

                def extract_text(entry):
                    try:
                        data = ast.literal_eval(entry)  # Safely convert string to dict
                        return data.get("text", "")  # Remove asterisks
                    except:
                        return ""

                df['clean_text'] = df['Comment Text'].apply(extract_text)
                with st.container(border=True):

                    st.subheader('Comments Word Cloud')
                    plotWordCloud(df)
                    # Step 2: Create a word cloud
                    # all_text = ' '.join(df['clean_text'].dropna())
                    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

                    # plt.figure(figsize=(10, 5))
                    # plt.imshow(wordcloud, interpolation='bilinear')
                    # plt.axis('off')
                    # plt.title("Word Cloud of Comments")
                    # st.pyplot(plt.gcf())
                    # # plt.show()

                # Step 3: Sentiment analysis using TextBlob
                def get_sentiment(text):
                    return TextBlob(text).sentiment.polarity  # Range: -1 (negative) to 1 (positive)
                with st.container(border=True):

                    st.subheader(f'Average Sentiment of Comments per {view_mode}')
                    st.caption("Negative Sentiment indicate meaner comments, positive indicates not as mean comments")
                    df['sentiment'] = df['clean_text'].apply(get_sentiment)
                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        watch_counts = df.groupby(['month'])['sentiment'].mean().reset_index(name='count')
                        watch_counts = watch_counts.sort_values('month')
                        # print(watch_counts)
                        # st.bar_chart(watch_counts.set_index('month'),color="#C4ED3B")  # can also use .bar_chart() 
                        plotCommentChart(watch_counts,time_col='month',chartType='Sent')

                    else:
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year'])['sentiment'].mean().reset_index(name='count')
                        
                        # st.bar_chart(watch_counts.set_index('year'),color="#C4ED3B")
                        plotCommentChart(watch_counts,time_col='year',chartType='Sent')
            
            elif tabControl=="Subsctripions & Playlists":
                st.markdown('---')

                    # st.subheader(f"Total Number of Subscriptions: {len(st.session_state['subs'])}")
                st.header("Playlists & Subscriptions")
                with st.container(border=True):

                    stats=st.columns(3)
                    with stats[0]:
                        ui.metric_card("Total Number of Subscriptions",f"{len(st.session_state['subs'])}")
                    with stats[1]:
                        ui.metric_card(f"Total Number of Playlists", f"{(len(st.session_state['Playlists']))}")
                    # st.subheader(f"Total Number of Playlists {(len(st.session_state['Playlists']))}")
                    with stats[2]:
                        ui.metric_card(f"Total Videos in All Playlists",f"{len(st.session_state['all_lists'])}")
                    # st.subheader(f'Total Videos in All Playlists {len(st.session_state["all_lists"])}')
                lengths=[]
                i=0

                for x in st.session_state['list_names']:
                    # if x ==st.session_state['list_names'][0]:
                    #     i+=1
                    #     continue
                    # else:
                    lengths.append(len(st.session_state['Playlists'][i]))
                    i+=1
                with st.container(border=True):

                    st.subheader("Videos per Playlist")
                    st.session_state['list_names']=[g.replace("Takeout/YouTube and YouTube Music/playlists/","") for g in st.session_state['list_names']]
                    # st.bar_chart(pd.DataFrame({"Playlist":st.session_state['list_names'],'Number of Videos':lengths}),x='Playlist',y='Number of Videos',color="#4DD799")
                    plotPlaylists(st.session_state["list_names"],lengths)
                df=st.session_state['all_lists']
                # df = st.session_state.history[((pd.to_datetime(stjj.session_state['all_lists']["Playlist Video Creation Timestamp"])).dt.to_period('M').dt.to_timestamp() >= pd.to_datetime(start_date).date()) & (pd.to_datetime(st.session_state['all_lists']["Playlist Video Creation Timestamp"]) <= pd.to_datetime(end_date).date())]

                df["Date"]=pd.to_datetime(df['Playlist Video Creation Timestamp'])
                df["Date2"]=df["Date"].dt.date
                df=df[(df["Date2"]>=pd.to_datetime(start_date).date()) & (df["Date2"]<=pd.to_datetime(end_date).date())]
                # print(df["month"])
                with st.container(border=True):

                    st.subheader(f"Videos Added to Playlist per {view_mode}")
                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        watch_counts = df.groupby(['month']).size().reset_index(name='count')
                        watch_counts = watch_counts.sort_values('month')
                        # print(watch_counts)
                        # st.bar_chart(watch_counts.set_index('month'),color="#FF1988")  # can also use .bar_chart() 
                        plotCommentChart(watch_counts,time_col='month',chartType='List')

                    else:
                        df['year'] = df['Date'].dt.year
                        watch_counts = df.groupby(['year']).size().reset_index(name='count')
                        # st.bar_chart(watch_counts.set_index('year'),color="#FF1988")
                        plotCommentChart(watch_counts,time_col='year',chartType='List')

                    # btn(username="robertmundo", floating=False, width=100)\
            elif tabControl=="AI":
                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                st.markdown('---')
                st.sidebar()
                with st.container(border=True):

                    st.subheader("Chat with your Data!")
                    # st.badge("AI",icon='🤖',color="violet",width='stretch')
                    
                    agent=createDocumentAgent(df)
                    # agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True,verbose=False)

                    user_input = st.text_input("Ask About Your Watch History",placeholder="Enter question")

                    if user_input:
                        with st.spinner("Thinking..."):
                            try:
                                response = agent.run(user_input)
                                st.session_state.chat_history.append(("You", user_input))
                                st.session_state.chat_history.append(("WatchBot", response))
                            except Exception as e:
                                st.error(f"Error: {e}")

                    # Display chat history
                    for sender, message in st.session_state.chat_history:
                        st.markdown(f"**{sender}:** {message}")

            #     st.header("Watch History")

    footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {

left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with 💖 by <a style='display: block; text-align: center;' href="https://www.robertmundo.netlify.app/" target="_blank">RPM III</a></p>
<style>.bmc-button img{width: 27px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{line-height: 36px !important;height:37px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#FF813F !important;border-radius: 3px !important;border: 1px solid transparent !important;padding: 1px 9px !important;font-size: 23px !important;letter-spacing: 0.6px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Cookie', cursive !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;-o-transition: 0.3s all linear !important;-webkit-transition: 0.3s all linear !important;-moz-transition: 0.3s all linear !important;-ms-transition: 0.3s all linear !important;transition: 0.3s all linear !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Cookie" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/robertmundo"><img src="https://www.buymeacoffee.com/assets/img/BMC-btn-logo.svg" alt="Buy me a coffee"><span style="margin-left:5px">Buy me a coffee</span></a>
</div>
"""
    st.markdown(footer,unsafe_allow_html=True)

