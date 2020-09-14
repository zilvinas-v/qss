import random
import math
import time
from qiskit import *

class MonoImage:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.data = []
		for y in range(height):
			self.data.append([0] * width)

def load_image(path, output):
	from PIL import Image
	image = Image.open(path)
	image_pixels = image.load()
	for y in range(output.height):
		for x in range(output.width):
			c = image_pixels[x, y][0]
			output.data[y][x] = c

def save_image(name, input):
	from PIL import Image
	image = Image.new("RGB", (input.width, input.height))
	image_pixels = image.load()
	for y in range(input.height):
		for x in range(input.width):
			c = input.data[y][x]
			image_pixels[x, y] = (c, c, c)

	image.save(name + '.png', "PNG")

def sample_pattern(x, y, width, height):
	nx = (x + 0.5) / width
	ny = (y + 0.5) / height

	def sample_wave(ny, frequency):
		return (-math.sin(ny * math.pi * 2 * frequency) + 1.0) / 2

	intensity = 0.0
	if ny < 0.75:
		if nx < 0.5:
			intensity = sample_wave(ny, 2)
		else:
			intensity = sample_wave(ny, 4)
	else:
		intensity = (nx // (1 / 5)) / 4

	return intensity

def shader(tx, ty, qc, qx, qy, condition, n_to_flip):
	assert(qx.size == qy.size)
	assert(qx.size == 1)
	assert(qy.size == 1)

	total_dim_bits = qx.size + qy.size

	# TODO: Have a more optimal certain bit count flipping procedure
	def not_bits(i):
		power_of_two = 1
		for j in range(total_dim_bits):
			if (i & power_of_two) != 0:
				q = None
				if j < qx.size:
					q = qx[j]
				else:
					q = qy[j - qx.size]
				qc.x(q)
			power_of_two <<= 1

	for i in range(n_to_flip):
		not_bits(i)

		qc.barrier()

		qc.h(qx[0])
		qc.ccx(condition, qy[0], qx[0])
		qc.h(qx[0])

		qc.barrier()

		not_bits(i)

def grover(qc, qx, qy, condition):
	assert(qx.size == qy.size)
	assert(qx.size == 1)
	assert(qy.size == 1)

	qc.h(qx)
	qc.h(qy)

	qc.x(qx)
	qc.x(qy)

	qc.barrier()

	# NOTE: This part would need to change if we had more dimension bits (only works with 1 per each)
	qc.h(qx[0])
	qc.ccx(condition, qy[0], qx[0])
	qc.h(qx[0])

	qc.barrier()

	qc.x(qx)
	qc.x(qy)

	qc.h(qx)
	qc.h(qy)

def qft(qc, qr):
	n = qr.size
	if n == 0:
		return

	for i in range(n):
		top = n - 1 - i
		qc.h(qr[top])
		for j in range(top):
			qc.cu1(math.pi / 2**j, qr[top - 1 - j], qr[top])

	# NOTE: Reverse the bits. The QSS paper seems to do this, doesn't work without it.
	for i in range(n // 2):
		qc.swap(qr[i], qr[n - 1 - i])

# NOTE: Returns a color in [0, 255] range
def map_count_to_intensity(count, counter_bits, tile_dim):
	norm_count = count / (1 << counter_bits)
	norm_hits = (math.cos(norm_count * 2.0 * math.pi) + 1) / 2.0
	hits = round(norm_hits * (tile_dim * tile_dim))
	c = round(norm_hits * 255)
	return c

def build_qss_circuit(tx, ty, tiles_in_full, tile_dim_bits, counter_bits, full_image):
	qx = QuantumRegister(tile_dim_bits)
	qy = QuantumRegister(tile_dim_bits)
	qcounter = QuantumRegister(counter_bits)
	cr = ClassicalRegister(counter_bits)
	qc = QuantumCircuit(qx, qy, qcounter, cr)

	qc.h(qx)
	qc.h(qy)
	qc.h(qcounter)

	tile_dim = 1 << tile_dim_bits

	total_sum = 0
	for ry in range(tile_dim):
		for rx in range(tile_dim):
			x = tx * tile_dim + rx
			y = ty * tile_dim + ry
			total_sum += full_image.data[y][x]

	target_intensity = (total_sum / (tile_dim * tile_dim)) / 255
	n_to_flip = round((tile_dim * tile_dim) * target_intensity)

	for i in range(counter_bits):
		for j in range(1 << i):
			qc.barrier()
			shader(tx, ty, qc, qx, qy, qcounter[i], n_to_flip)
			qc.barrier()
			grover(qc, qx, qy, qcounter[i])

	qc.barrier()
	qft(qc, qcounter)
	qc.barrier()

	for i in range(counter_bits):
		qc.measure(qcounter[i], cr[i])

	return qc

def print_error(image, ideal_image):
	assert(ideal_image.width == image.width)
	assert(ideal_image.height == image.height)

	total_pixels = image.width * image.height
	total_pixels_color = total_pixels * 255

	total_error_free_pixels = 0
	total_error = 0

	for y in range(image.height):
		for x in range(image.width):
			pixel = image.data[y][x]
			ideal = ideal_image.data[y][x]

			if pixel == ideal:
				total_error_free_pixels += 1

			total_error += abs(ideal - pixel)

	print('    average pixel error: ', total_error / total_pixels_color * 100.0)
	print('    error free pixels: ', total_error_free_pixels / total_pixels * 100.0)

def generate_full_image(output, tiles_in_full, tile_dim):
	for ty in range(tiles_in_full):
		for tx in range(tiles_in_full):
			intensity = sample_pattern(tx, ty, tiles_in_full, tiles_in_full)
			fill_count = round((tile_dim * tile_dim) * intensity)

			for i in range(fill_count):
				rx = i % tile_dim
				ry = i // tile_dim
				x = tx * tile_dim + rx
				y = ty * tile_dim + ry
				output.data[y][x] = 255

def generate_ideal_image(output, input, tiles_in_full, tile_dim):
	for ty in range(tiles_in_full):
		for tx in range(tiles_in_full):
			sum = 0
			for ry in range(tile_dim):
				for rx in range(tile_dim):
					x = tx * tile_dim + rx
					y = ty * tile_dim + ry
					sum += input.data[y][x]

			average = sum // (tile_dim * tile_dim)
			output.data[ty][tx] = average

def generate_monte_carlo_image(output, input, sample_count, tiles_in_full, tile_dim):
	for ty in range(tiles_in_full):
		for tx in range(tiles_in_full):
			sum = 0
			for i in range(sample_count):
				rx = math.floor(random.random() * tile_dim)
				ry = math.floor(random.random() * tile_dim)
				x = tx * tile_dim + rx
				y = ty * tile_dim + ry
				sum += input.data[y][x]

			average = sum // sample_count
			output.data[ty][tx] = average

def generate_qss_image(output, input, tiles_in_full, tile_dim, tile_dim_bits, counter_bits, backend):
	circuits = []

	for ty in range(tiles_in_full):
		for tx in range(tiles_in_full):
			qc = build_qss_circuit(tx, ty, tiles_in_full, tile_dim_bits, counter_bits, input)
			circuits.append(qc)

	results = []

	# NOTE: This is assumed to divide the number of tiles evenly.
	# Also, backends have limits to how many you can have per batch.
	jobs_per_batch = 64

	for i in range(len(circuits) // jobs_per_batch):
		start = i * jobs_per_batch
		end = start + jobs_per_batch
		print(start, end)
		experiments = circuits[start : end]
		result = execute(experiments, backend, shots=1).result()
		results.append(result)

	total_time = 0

	for i in range(len(results)):
		result = results[i]
		total_time += result.time_taken

		for j in range(jobs_per_batch):
			tx = (i * jobs_per_batch + j) % tiles_in_full
			ty = (i * jobs_per_batch + j) // tiles_in_full

			qc = circuits[i * jobs_per_batch + j]
			counts = result.get_counts(qc)

			counter = 0
			max_counter_count = 0
			for key in counts.keys():
				count = counts[key]
				if count > max_counter_count:
					max_counter_count = count
					counter = int(key, 2)

			c = map_count_to_intensity(counter, counter_bits, tile_dim)
			output.data[ty][tx] = c

	return total_time

def run_qss_experiments(name, dim_bits, tile_dim_bits, counter_bits, do_qss_sim = True, do_qss_real = True):
	full_dim = 1 << dim_bits
	tile_dim = 1 << tile_dim_bits
	ss_dim = full_dim // tile_dim
	tiles_in_full = full_dim // tile_dim

	IBMQ.load_account()
	provider = IBMQ.get_provider('ibm-q')

	simulator = Aer.get_backend('qasm_simulator')
	# simulator = provider.get_backend('ibmq_qasm_simulator')

	real_computer = provider.get_backend('ibmq_ourense')

	print('generating the full image...')
	full_image = MonoImage(full_dim, full_dim)
	generate_full_image(full_image, tiles_in_full, tile_dim)

	print('generating the ideal image...')
	ideal_image = MonoImage(ss_dim, ss_dim)
	generate_ideal_image(ideal_image, full_image, tiles_in_full, tile_dim)

	print('generating the monte carlo image...')
	monte_sample_count = (1 << counter_bits) - 1
	monte_image = MonoImage(ss_dim, ss_dim)
	generate_monte_carlo_image(monte_image, full_image, monte_sample_count, tiles_in_full, tile_dim)

	qss_sim_image = MonoImage(ss_dim, ss_dim)
	qss_sim_total_time = 0
	if do_qss_sim:
		print('generating the qss sim image...')
		qss_sim_total_time = generate_qss_image(qss_sim_image, full_image, tiles_in_full, tile_dim, tile_dim_bits, counter_bits, simulator)

	qss_real_image = MonoImage(ss_dim, ss_dim)
	qss_real_total_time = 0
	if do_qss_real:
		print('generating the qss real image...')
		qss_real_total_time = generate_qss_image(qss_real_image, full_image, tiles_in_full, tile_dim, tile_dim_bits, counter_bits, real_computer)

	print('--- results ---')
	save_image(name + '_full', full_image)
	save_image(name + '_ideal', ideal_image)

	save_image(name + '_monte', monte_image)
	print('monte samples: ', monte_sample_count)
	print('monte error:')
	print_error(monte_image, ideal_image)

	print(f"qss parameters: tile_dimension_bits = {tile_dim_bits}, counter_bits = {counter_bits}")

	if do_qss_sim:
		save_image(name + '_qss_sim', qss_sim_image)
		print('qss sim total time: ', qss_sim_total_time)
		print('qss sim error:')
		print_error(qss_sim_image, ideal_image)

	if do_qss_real:
		save_image(name + '_qss_real', qss_real_image)
		print('qss real total time: ', qss_real_total_time)
		print('qss real error:')
		print_error(qss_real_image, ideal_image)

# run_qss_experiments('exp1', 8, 1, 2, do_qss_sim = True, do_qss_real = False)
run_qss_experiments('exp2', 8, 1, 1, do_qss_sim = True, do_qss_real = False)
