import core

print("Testing core.py processing...")
res1 = core.process_audio('test_audio/audio1.wav')
res2 = core.process_audio('test_audio/audio2.wav')
res3 = core.process_audio('test_audio/audio3.wav')

print("Audio1:", res1.get('error', 'No Error'), {k:v for k,v in res1.items() if k != "embeddings"})
print("Audio3 (short):", {k:v for k,v in res3.items() if k != "embeddings"})

sim12 = core.compute_similarity(res1['embeddings'], res2['embeddings'])
sim13 = core.compute_similarity(res1['embeddings'], res3['embeddings'])

print("Similarity 1 vs 2:", sim12)
print("Similarity 1 vs 3:", sim13)

penalties, violations = core.calculate_transitivity([[1.0, sim12, sim13], [sim12, 1.0, 0.2], [sim13, 0.2, 1.0]], ["audio1", "audio2", "audio3"])
print("Transitivity violations:", violations)
