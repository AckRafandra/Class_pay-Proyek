from flask import Flask, jsonify
from flask import Request, Response
import os

# Membuat aplikasi Flask
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(message="Hello from Flask on Netlify!")

def handler(event, context):
    """ Fungsi untuk menjalankan Flask dalam serverless function """
    # Membuat objek request dan response untuk flask
    from werkzeug.wrappers import Request, Response

    # Mengubah event menjadi objek request
    request = Request(event)

    # Menggunakan Flask app untuk menangani request
    with app.request_context(request):
        response = app.full_dispatch_request()
    
    # Kembalikan response
    return {
        'statusCode': response.status_code,
        'body': response.get_data(as_text=True),
        'headers': {'Content-Type': 'application/json'}
    }
