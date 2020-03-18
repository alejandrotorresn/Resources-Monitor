import psutil
import pynvml
import getpass
import pandas as pd
import time

KB = 1e3
MB = KB * KB
GB = MB * KB

pynvml.nvmlInit()
ngpus = pynvml.nvmlDeviceGetCount()
gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(ngpus)]

def find_procs_by_name(name, name_exe):
	indexs = ['pid', 'name', 'started', 'cpu_percent', 'memory_percent']
	ls = [0, 0, 0, 0, 0]
	x = pd.Series(dtype=object)
	for p in psutil.process_iter(['pid', 'name', 'username', 'cmdline', 'status']):
		if len(p.info['cmdline']) == 2:
			if p.info['name'] == name  and p.info['cmdline'][1] == name_exe:
				ls = [p.ppid(), p.name(), p.create_time(), p.cpu_percent(), p.memory_percent()]
				x = pd.Series(ls, index=indexs)
				break
	return x

def get_gpu_utilization():
	return [
		pynvml.nvmlDeviceGetUtilizationRates(gpu_handles[i]).gpu for i in range(ngpus)
		]
	
def gpu_mem():
	return [pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in gpu_handles]

def gpu_mem_total():
	return 	pynvml.nvmlDeviceGetMemoryInfo(gpu_handles[0]).total

def pci():
	pci_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(gpu_handles[0])
	pci_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(gpu_handles[0])
	pci_bw = {
		1: (250.0 * MB),
		2: (500.0 * MB),
		3: (985.0 * MB),
		4: (1969.0 * MB),
		5: (3938.0 * MB),
		6: (7877.0 * MB),
	}

	max_rxtx_tp = pci_width * pci_bw[pci_gen]
	
	pci_tx = [
		pynvml.nvmlDeviceGetPcieThroughput(gpu_handles[i], pynvml.NVML_PCIE_UTIL_TX_BYTES) * KB for i in range(ngpus)
	]

	pci_rx = [
		pynvml.nvmlDeviceGetPcieThroughput(gpu_handles[i], pynvml.NVML_PCIE_UTIL_RX_BYTES) * KB for i in range(ngpus)
	]

	indexs = ['max_rxtx_tp', 'pci_tx', 'pci_rx']
	pci = pd.Series([max_rxtx_tp, pci_tx, pci_rx], index=indexs)

	return pci
	

def process_data_capture(process_name, name_exe):

	df1 = pd.Series(dtype=object)
	indexs = ['time','cpu_percent','memory_percent','gpu_utlizacion','gpu_memory_utilization', 'pcie_tx', 'pcie_rx']

	if not find_procs_by_name(process_name, name_exe).empty:
		
		start = time.time()
		
		cpu_percent = find_procs_by_name(process_name, name_exe)[3]
		memory_percent = find_procs_by_name(process_name, name_exe)[4]
		gpu_utilization = get_gpu_utilization()
		gpu_memory_utilization = gpu_mem()[0]/MB
		pcie_tx = pci()[1][0] 
		pcie_rx = pci()[2][0]

		df1 = pd.Series([start, cpu_percent, memory_percent, gpu_utilization[0], gpu_memory_utilization, pcie_tx, pcie_rx], index=indexs)
		
		return df1


def process_info():
	started = find_procs_by_name(process_name, name_exe)[2]
	gpu_memory_T = gpu_mem_total()/MB
	pci_width = pci()[0][0]/MB
	

def main():

	process_name = 'python' 
	name_exe = 'nn_PyTorch.py'	
	df = pd.DataFrame(dtype=object)
	if ngpus > 0:
		try:
			while(True):
				dp = process_data_capture(process_name, name_exe)
				df = df.append(dp, ignore_index=True)
				if not df.empty:
					print(df)
		except KeyboardInterrupt:
			df.to_csv('reports/'+ name_exe +'_report.csv', index=False)
			pass

if __name__ == "__main__":
    main()

