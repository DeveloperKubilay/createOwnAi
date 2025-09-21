
import os
import io
import sys
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Google Drive API yetki kapsamı
SCOPES = ['https://www.googleapis.com/auth/drive.file']

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
	media = MediaFileUpload(file_path, mimetype=mime_type)
	file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
	print(f"Yüklendi! Dosya ID: {file.get('id')}")
	return file.get('id')

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

if __name__ == "__main__":
	print("Google Drive bağlantısı için tarayıcı açılacak. İlk defa çalıştırıyorsan Google hesabınla giriş yap!")
	if len(sys.argv) >= 3:
		cmd = sys.argv[1]
		arg = sys.argv[2]
		if cmd == "upload":
			if not os.path.exists(arg):
				print(f"Dosya bulunamadı: {arg}")
				sys.exit(1)
			# MIME type otomatik belirlenmeye çalışılır, bilinmiyorsa octet-stream kullanılır
			import mimetypes
			mime_type, _ = mimetypes.guess_type(arg)
			if not mime_type:
				mime_type = 'application/octet-stream'
			upload_file(arg, mime_type)
		elif cmd == "download":
			# Burada arg drive dosya ID'si olmalı, 3. argüman varsa hedef dosya adı olarak kullanılır
			if len(sys.argv) >= 4:
				dest = sys.argv[3]
			else:
				dest = arg
			download_file(arg, dest)
		else:
			print("Kullanım: python main.py upload <dosya_yolu> veya python main.py download <drive_file_id> [hedef_dosya]")
	else:
		print("Kullanım: python main.py upload <dosya_yolu> veya python main.py download <drive_file_id> [hedef_dosya]")
