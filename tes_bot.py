def get_live_price(symbol):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url).json()
        if 'message' in response:
            return f"Server: {response['message']}"
        return response.get('price', 'Data tidak ditemukan')
    except Exception as e:
        return f"Error: {str(e)}"
