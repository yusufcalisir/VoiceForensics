import core

res = core.process_audio('test_audio/audio1.wav')
print("--- ERROR IS ---")
print(res.get("error"))
