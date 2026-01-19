import { exists } from 'https://deno.land/std@0.224.0/fs/mod.ts';
import { join } from 'https://deno.land/std@0.224.0/path/mod.ts';

import { arch as getArch, cpus, platform as getPlatform } from 'node:os';

import { Command, EnumType } from '@cliffy/command';
import $ from '@david/dax';

const arch = getArch() as 'x64' | 'arm64';
const platform = getPlatform() as 'win32' | 'darwin' | 'linux';

const TARGET_ARCHITECTURE_TYPE = new EnumType([ 'x86_64', 'aarch64' ]);

function getCudnnArchiveUrl(arch: string): string {
    if (platform === 'linux') {
        return arch === 'aarch64'
            ? 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-aarch64/cudnn-linux-aarch64-9.10.0.56_cuda12-archive.tar.xz'
            : 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.10.0.56_cuda12-archive.tar.xz';
    } else {
        return 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.0.56_cuda12-archive.zip';
    }
}

const TENSORRT_ARCHIVE_URL = platform === 'linux'
	? 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.10.0/tars/TensorRT-10.10.0.31.Linux.x86_64-gnu.cuda-12.9.tar.gz'
	: 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.10.0/zip/TensorRT-10.10.0.31.Windows.win10.cuda-12.9.zip';
const TENSORRT_RTX_ARCHIVE_URL = platform === 'linux'
	? 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.1/TensorRT-RTX-1.1.1.26.Linux.x86_64-gnu.cuda-12.9.tar.gz'
	: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.1/TensorRT-RTX-1.1.1.26.Windows.win10.cuda-12.9.zip';

await new Command()
	.name('ort-artifact')
	.version('0.1.0')
	.type('target-arch', TARGET_ARCHITECTURE_TYPE)
	.option('-v, --upstream-version <version:string>', 'Exact version of upstream package', { required: true })
	.option('-t, --training', 'Enable Training API')
	.option('-s, --static', 'Build static library')
	.option('--iphoneos', 'Target iOS / iPadOS')
	.option('--iphonesimulator', 'Target iOS / iPadOS simulator')
	.option('--android', 'Target Android')
	.option('--cuda', 'Enable CUDA EP')
	.option('--trt', 'Enable TensorRT EP', { depends: [ 'cuda' ] })
	.option('--nvrtx', 'Enable NV TensorRT RTX EP')
	.option('--directml', 'Enable DirectML EP')
	.option('--coreml', 'Enable CoreML EP')
	.option('--dnnl', 'Enable DNNL EP')
	.option('--xnnpack', 'Enable XNNPACK EP')
	.option('--webgpu', 'Enable WebGPU EP')
	.option('--openvino', 'Enable OpenVINO EP')
	.option('--nnapi', 'Enable NNAPI EP')
	.option('-N, --ninja', 'build with ninja')
	.option('-A, --arch <arch:target-arch>', 'Configure target architecture for cross-compile', { default: 'x86_64' })
	.action(async (options, ..._) => {
		const root = Deno.cwd();

		const onnxruntimeRoot = join(root, 'onnxruntime');
		const isExists = await exists(onnxruntimeRoot)
		let isBranchCorrect = false;
		if (isExists) {
			$.cd(onnxruntimeRoot);
			const currentBranch = (await $`git branch --show-current`.stdout("piped")).stdout.trim()
			isBranchCorrect = currentBranch === `rel-${options.upstreamVersion}`;
			$.cd(root);
			
			if (!isBranchCorrect) {
				console.log(`Removing onnxruntime directory because branch is incorrect: ${onnxruntimeRoot}, current branch: ${currentBranch}, expected branch: rel-${options.upstreamVersion}`);
				await Deno.remove(onnxruntimeRoot, { recursive: true });
			}
		}
		if (!isExists || !isBranchCorrect) {
			await $`git clone https://github.com/microsoft/onnxruntime --recursive --single-branch --depth 1 --branch rel-${options.upstreamVersion}`;
		}

		$.cd(onnxruntimeRoot);

		await $`git reset --hard HEAD`;
		await $`git clean -fdx`;

		const patchDir = join(root, 'src', 'patches', 'all');
		for await (const patchFile of Deno.readDir(patchDir)) {
			if (!patchFile.isFile) {
				continue;
			}

			await $`git apply ${join(patchDir, patchFile.name)} --ignore-whitespace --recount --verbose`;
			console.log(`applied ${patchFile.name}`);
		}

		const env = { ...Deno.env.toObject() };
		const args = [];
		const compilerFlags = [];
		const cudaFlags: string[] = [];
		// [NEW] 定义默认 toolchain 文件名
		let crossToolchainFile = 'aarch64-unknown-linux-gnu.cmake';
		
		if (platform === 'linux' && !options.android) {
			// env.CC = 'clang-18';
			// env.CXX = 'clang++-18';
			// if (options.cuda) {
			// 	cudaFlags.push('-ccbin', 'clang++-18');
			// }

			// new code - 2026/01
			if (options.arch === 'aarch64') {
				// Linux Cross-compile (x64 -> aarch64)
				
				if (options.cuda) {
					// 1. 自动探测存在的 aarch64 g++ 编译器版本
					const possibleCompilers = [
						'aarch64-linux-gnu-g++',    // 尝试通用名
						'aarch64-linux-gnu-g++-11',
						'aarch64-linux-gnu-g++-12',
						'aarch64-linux-gnu-g++-13',
						'aarch64-linux-gnu-g++-14',
					];
					
					let hostCompiler = '';
					for (const compiler of possibleCompilers) {
						try {
							await $`which ${compiler}`.quiet();
							hostCompiler = compiler;
							console.log(`Found CUDA host compiler: ${hostCompiler}`);
							break;
						} catch {
							continue;
						}
					}

					if (!hostCompiler) {
						console.error("Warning: Could not find aarch64-linux-gnu-g++. CUDA build may fail.");
						hostCompiler = 'aarch64-linux-gnu-g++';
					}
					// [NEW] 如果检测到 gcc-13，切换 toolchain 文件
					if (hostCompiler.includes('g++-13')) {
						crossToolchainFile = 'aarch64-unknown-linux-gnu-gcc13.cmake';
					}
                    // 设置 CMake 变量以使用正确的宿主编译器进行链接
                    args.push(`-DCMAKE_CUDA_HOST_COMPILER=${hostCompiler}`);

					// 2. 关键修复：修补 CUDA 安装以支持交叉编译
                    // GitHub Runner 的 CUDA 通常缺少 targets/aarch64-linux 目录
					try {
						const nvccPath = (await $`which nvcc`.text()).trim();
						if (nvccPath) {
                            const cudaPath = join(nvccPath, '..', '..');
                            const targetProfileDir = join(cudaPath, 'targets', 'aarch64-linux');
                            
                            // 2a. 修复头文件路径 (解决 fatal error: cuda_runtime.h: No such file)
                            if (!(await exists(join(targetProfileDir, 'include')))) {
                                console.log("Patching CUDA: Creating target include directory symlink...");
                                await $`sudo mkdir -p ${targetProfileDir}`;
                                // 将 include 链接到通用的 include 目录
                                await $`sudo ln -sf ../../include ${join(targetProfileDir, 'include')}`;
                            }

                            // 2b. 修复库文件路径 (解决链接错误)
                            // 下载 NVIDIA 官方的 cudart (aarch64) 并放入 lib 目录
                            const libDir = join(targetProfileDir, 'lib');
                            if (!(await exists(libDir))) {
                                console.log("Patching CUDA: Downloading aarch64 cudart libraries...");
                                const tmpDir = await Deno.makeTempDir();
                                // 使用与 CUDA 12.x 兼容的 cudart 版本 (12.8.57)
                                const cudartUrl = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-aarch64/cuda_cudart-linux-aarch64-12.8.57-archive.tar.xz';
                                const tarPath = join(tmpDir, 'cudart.tar.xz');
                                
                                await $`curl -L -o ${tarPath} ${cudartUrl}`;
                                await $`tar -xf ${tarPath} -C ${tmpDir}`;
                                
                                // 提取库文件
                                const extractedLib = join(tmpDir, 'cuda_cudart-linux-aarch64-12.8.57-archive', 'lib');
                                await $`sudo mkdir -p ${libDir}`;
                                await $`sudo cp -r ${extractedLib}/. ${libDir}/`;
                                
                                // 清理
                                await Deno.remove(tmpDir, { recursive: true });
                            }
                            
                            // 告诉 CMake 库文件的位置 (虽然 nvcc 会自动查找 targets/aarch64-linux/lib，但显式指定更安全)
                            args.push(`-DCMAKE_CUDA_COMPILER_LIBRARY_ROOT=${targetProfileDir}`);
						}
					} catch (e) {
						console.warn("Error patching CUDA environment:", e);
					}
				}
			} else {
				// Linux Native (x64)
				env.CC = 'clang-18';
				env.CXX = 'clang++-18';
				if (options.cuda) {
					cudaFlags.push('-ccbin', 'clang++-18');
				}
			}
			// new ode -end-
			
		} else if (platform === 'win32') {
			args.push('-G', 'Visual Studio 17 2022');
			if (options.arch === 'x86_64') {
				args.push('-A', 'x64');
			}
		}

		// Build for iOS on macOS.
		if (platform === 'darwin' && (options.iphoneos || options.iphonesimulator)) {
			args.push(`-DCMAKE_OSX_DEPLOYMENT_TARGET=${Deno.env.get("IPHONEOS_DEPLOYMENT_TARGET")}`)
			args.push('-DCMAKE_TOOLCHAIN_FILE=../cmake/onnxruntime_ios.toolchain.cmake');
			if(options.iphoneos) {
				args.push('-DCMAKE_OSX_SYSROOT=iphoneos');
			} else {
				args.push('-DCMAKE_OSX_SYSROOT=iphonesimulator');
			}
		}

		// Build for Android on Linux.
		if (platform === 'linux' && options.android) {
			// ANDROID_NDK_HOME and ANDROID_SDK_ROOT are expected to be set in the environment.
			args.push(`-DANDROID_PLATFORM=android-${Deno.env.get("ANDROID_API")}`);
			args.push('-DANDROID_ABI=arm64-v8a');
			args.push('-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false');
			args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(Deno.env.get('ANDROID_NDK_HOME')!, 'build', 'cmake', 'android.toolchain.cmake')}`);
		}

		if (options.cuda) {
			args.push('-Donnxruntime_USE_CUDA=ON');
			// https://github.com/microsoft/onnxruntime/pull/20768
			args.push('-Donnxruntime_NVCC_THREADS=1');

			const cudnnOutPath = join(root, 'cudnn');
			let should_skip = await exists(cudnnOutPath);
			if (should_skip) {
				// Check dir whether is empty
				const files = await Array.fromAsync(Deno.readDir(cudnnOutPath));
				if (files.length === 0) {
					await $`rm -rf ${cudnnOutPath}`;
					should_skip = false;
				}
			}

			if (!should_skip) {
				const cudnnArchiveUrl = getCudnnArchiveUrl(options.arch);
				const cudnnArchiveStream = await fetch(cudnnArchiveUrl).then(c => c.body!);
				await Deno.mkdir(cudnnOutPath);
				await $`tar xvJC ${cudnnOutPath} --strip-components=1 -f -`.stdin(cudnnArchiveStream);
			}
			
			args.push(`-Donnxruntime_CUDNN_HOME=${cudnnOutPath}`);

			if (platform === 'win32') {
				// nvcc < 12.4 throws an error with VS 17.10
				cudaFlags.push('-allow-unsupported-compiler');
			}
		}

		if (options.cuda || options.trt || options.nvrtx) {
			args.push('-Donnxruntime_USE_FPA_INTB_GEMM=OFF');
			const cudaArchs = options.arch === 'aarch64' ? '72;87' : '75;80;90';
    		args.push(`-DCMAKE_CUDA_ARCHITECTURES=${cudaArchs}`);
			cudaFlags.push('-compress-mode=size');
		}

		if (options.trt) {
			args.push('-Donnxruntime_USE_TENSORRT=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
		}
		if (options.nvrtx) {
			args.push('-Donnxruntime_USE_NV=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
		}

		if (options.trt) {
			const trtArchiveStream = await fetch(TENSORRT_ARCHIVE_URL).then(c => c.body!);
			const trtOutPath = join(root, 'tensorrt');
			await Deno.mkdir(trtOutPath);
			await $`tar xvzC ${trtOutPath} --strip-components=1 -f -`.stdin(trtArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_HOME=${trtOutPath}`);
		}
		if (options.nvrtx) {
			const trtxArchiveStream = await fetch(TENSORRT_RTX_ARCHIVE_URL).then(c => c.body!);
			const trtxOutPath = join(root, 'tensorrt');
			await Deno.mkdir(trtxOutPath);
			await $`tar xvzC ${trtxOutPath} --strip-components=1 -f -`.stdin(trtxArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_RTX_HOME=${trtxOutPath}`);
		}

		if (platform === 'win32' && options.directml) {
			args.push('-Donnxruntime_USE_DML=ON');
		}
		if (platform === 'darwin' && options.coreml) {
			args.push('-Donnxruntime_USE_COREML=ON');
		}
		if (options.webgpu) {
			args.push('-Donnxruntime_USE_WEBGPU=ON');
			args.push('-Donnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS=OFF');
			args.push('-Donnxruntime_USE_EXTERNAL_DAWN=OFF');
			args.push('-Donnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY=ON');
			args.push('-Donnxruntime_WGSL_TEMPLATE=static')
		}
		if (options.dnnl) {
			args.push('-Donnxruntime_USE_DNNL=ON');
		}
		if (options.xnnpack) {
			args.push('-Donnxruntime_USE_XNNPACK=ON');
		}
		if (options.openvino) {
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
			args.push('-Donnxruntime_USE_OPENVINO=ON');
			args.push('-Donnxruntime_USE_OPENVINO_CPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_GPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_NPU=ON');
			// args.push('-Donnxruntime_USE_OPENVINO_INTERFACE=ON');
		}
		if(options.nnapi) {
			args.push('-Donnxruntime_USE_NNAPI_BUILTIN=ON');
		}

		if (platform === 'darwin') {
			if (options.arch === 'aarch64') {
				args.push('-DCMAKE_OSX_ARCHITECTURES=arm64');
			} else {
				args.push('-DCMAKE_OSX_ARCHITECTURES=x86_64');
			}
		} else {
			if (options.arch === 'aarch64' && arch !== 'arm64') {
				args.push('-Donnxruntime_CROSS_COMPILING=ON');
				switch (platform) {
					case 'win32':
						args.push('-A', 'ARM64');
						compilerFlags.push('-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS');
						break;
					case 'linux':
						if (!options.android) {
							args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(root, 'toolchains', crossToolchainFile)}`);
						}
						break;
				}
			}
		}

		if (options.training) {
			args.push('-Donnxruntime_ENABLE_TRAINING=ON');
			args.push('-Donnxruntime_ENABLE_LAZY_TENSOR=OFF');
		}

		if (options.training) {
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
		}

		if (platform === 'win32' && !options.static) {
			args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dgtest_force_shared_crt=ON');
		}

		if (!options.static) {
			args.push('-Donnxruntime_BUILD_SHARED_LIB=ON');
		} else {
			if (platform === 'win32') {
				args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dgtest_force_shared_crt=ON');
				args.push('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL');
			}
		}

		// https://github.com/microsoft/onnxruntime/pull/21005
		if (platform === 'win32') {
			compilerFlags.push('-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR');
		}

		args.push('-Donnxruntime_BUILD_UNIT_TESTS=OFF');
		args.push(`-Donnxruntime_USE_KLEIDIAI=${options.arch === 'aarch64' ? 'ON' : 'OFF'}`);
		args.push('-Donnxruntime_CLIENT_PACKAGE_BUILD=ON');

		if (options.arch === 'x86_64') {
			switch (platform) {
				case 'linux':
					compilerFlags.push('-march=x86-64-v3');
					break;
				case 'win32':
					// compilerFlags.push('/arch:AVX2');
					compilerFlags.push('-march=x86-64-v3');
					break;
			}
		}

		if (compilerFlags.length > 0) {
			const allFlags = compilerFlags.join(' ');
			args.push(`-DCMAKE_C_FLAGS=${allFlags}`);
			args.push(`-DCMAKE_CXX_FLAGS=${allFlags}`);
		}

		if (options.ninja && !(platform === 'win32' && options.arch === 'aarch64')) {
			args.push('-G', 'Ninja');
		}

		if (cudaFlags.length) {
			args.push(`-DCMAKE_CUDA_FLAGS_INIT=${cudaFlags.join(' ')}`);
		}

		const sourceDir = options.static ? join(root, 'src', 'static-build') : 'cmake';
		const artifactOutDir = join(root, 'artifact', 'onnxruntime');

		await $`cmake -S ${sourceDir} -B build -D CMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release -DCMAKE_INSTALL_PREFIX=${artifactOutDir} -DONNXRUNTIME_SOURCE_DIR=${onnxruntimeRoot} --compile-no-warning-as-error ${args}`
			.env(env);
		await $`cmake --build build --config Release --parallel ${cpus().length}`;
		await $`cmake --install build`;
	})
	.parse(Deno.args);
