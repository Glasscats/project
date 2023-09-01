from aip import AipSpeech

config = {
    'appId' : '38196884',
    'apiKey' : 'ffm7CTCRw2qcT0leV8m0gy1L',
    'secretKey' : 'lVXH3Dliz9OfxldZxaNbFnRI3DZ7h2Nw'
}

client = AipSpeech(**config)

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def Get_str_from_voice(voice_path):
    voice = get_file_content(voice_path)
    result = client.asr(voice, 'wav', 16000)
    print(result)
    #print('\n'.join(result.get('result')))

Get_str_from_voice("./test.wav")
