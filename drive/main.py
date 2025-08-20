
import os
import io
import sys
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from datetime import datetime
try:
	from zoneinfo import ZoneInfo
except Exception:
	ZoneInfo = None

# Google Drive API yetki kapsamı
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
	creds = None
	if os.path.exists('token.pickle'):
		with open('token.pickle', 'rb') as token:
			creds = pickle.load(token)
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			# client_secret dosyasının adını güncelledim
			flow = InstalledAppFlow.from_client_secrets_file('client_secret_528336115917-ugunu8o9mg1vrcnufmjla60h3rs6p9hu.apps.googleusercontent.com.json', SCOPES)
			creds = flow.run_local_server(port=0)
		with open('token.pickle', 'wb') as token:
			pickle.dump(creds, token)
	service = build('drive', 'v3', credentials=creds)
	return service

def upload_file(file_path, mime_type):
	service = get_drive_service()
	file_metadata = {'name': os.path.basename(file_path)}
	# 10MB chunk boyutu, RAM dostu
	chunk_size = 10 * 1024 * 1024
	media = MediaFileUpload(file_path, mimetype=mime_type, chunksize=chunk_size, resumable=True)
	request = service.files().create(body=file_metadata, media_body=media, fields='id')
	response = None
	while response is None:
		status, response = request.next_chunk()
		if status:
			print(f"Yükleniyor: %{int(status.progress() * 100)} tamamlandı.")
	print(f"Yüklendi! Dosya ID: {response.get('id')}")
	return response.get('id')

def download_file(file_id, destination):
	service = get_drive_service()
	request = service.files().get_media(fileId=file_id)
	fh = io.FileIO(destination, 'wb')
	downloader = MediaIoBaseDownload(fh, request)
	done = False
	while not done:
		status, done = downloader.next_chunk()
		print(f"İndirme: %{int(status.progress() * 100)} tamamlandı.")
	print(f"İndirildi! {destination}")

def list_files(page_size=100, q=None, include_all_drives=True):
	service = get_drive_service()
	request = service.files().list(pageSize=page_size,
		fields="nextPageToken, files(id, name, modifiedTime, shared)",
		q=q,
		includeItemsFromAllDrives=include_all_drives,
		supportsAllDrives=include_all_drives)
	results = request.execute()
	items = results.get('files', [])
	if not items:
		print('Hiç dosya bulunamadı.')
		return
	print(f"{'ID':40}  {'İsim':40}  {'Ortak':6}  {'Son Değiştirilme'}")
	for f in items:
		mod = f.get('modifiedTime')
		shared_flag = 'E' if f.get('shared') else 'H'
		if mod:
			try:
				if ZoneInfo:
					tz = ZoneInfo('Europe/Istanbul')
					dt = datetime.fromisoformat(mod.replace('Z', '+00:00')).astimezone(tz)
				else:
					dt = datetime.fromisoformat(mod.replace('Z', '+00:00')).astimezone()
				modstr = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
			except Exception:
				modstr = mod
		else:
			modstr = ''
		print(f"{f.get('id', ''):40}  {f.get('name', ''):40}  {shared_flag:6}  {modstr}")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Kullanım: python main.py upload <dosya_yolu> | download <drive_file_id> [hedef_dosya] | list [adet]")
		sys.exit(1)
	cmd = sys.argv[1]
	if cmd == "upload":
		if len(sys.argv) < 3:
			print("upload için dosya yolu gerekli: python main.py upload <dosya_yolu>")
			sys.exit(1)
		arg = sys.argv[2]
		if not os.path.exists(arg):
			print(f"Dosya bulunamadı: {arg}")
			sys.exit(1)
		import mimetypes
		mime_type, _ = mimetypes.guess_type(arg)
		if not mime_type:
			mime_type = 'application/octet-stream'
		upload_file(arg, mime_type)
	elif cmd == "download":
		if len(sys.argv) < 3:
			print("download için dosya ID gerekli: python main.py download <drive_file_id> [hedef_dosya]")
			sys.exit(1)
		arg = sys.argv[2]
		if len(sys.argv) >= 4:
			dest = sys.argv[3]
		else:
			dest = arg
		download_file(arg, dest)
	elif cmd == "list":
		# Usage: list [adet] | list shared [adet] | list all [adet]
		mode = None
		page_size = 100
		if len(sys.argv) >= 3:
			if sys.argv[2] in ("shared", "all"):
				mode = sys.argv[2]
				if len(sys.argv) >= 4:
					try:
						page_size = int(sys.argv[3])
					except ValueError:
						pass
			else:
				try:
					page_size = int(sys.argv[2])
				except ValueError:
					page_size = 100
		# decide query and include_all_drives
		q = None
		include_all = True
		if mode == 'shared':
			q = 'sharedWithMe = true' 
		elif mode == 'all':
			q = None
		list_files(page_size=page_size, q=q, include_all_drives=include_all)
	else:
		print("Kullanım: python main.py upload <dosya_yolu> | download <drive_file_id> [hedef_dosya] | list [adet]")
