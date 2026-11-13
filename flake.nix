{
  description = "tapefront - Live Nix Notetaker: CLI meeting recorder (mic + system audio) with a live transcript";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      py = pkgs.python3Packages;
      version = "0.4.1"; # transcribe.cpp release
      release = "https://github.com/handy-computer/transcribe.cpp/releases/download/v${version}";

      # Prebuilt native library + ggml CPU backends. The per-CPU libggml-cpu-*
      # variants stay side by side: ggml picks the best one at runtime. The
      # Vulkan backend (dlopen'ed, not linked) is left out: live is CPU-only.
      transcribe-native = pkgs.stdenv.mkDerivation {
        pname = "transcribe-native";
        inherit version;
        src = pkgs.fetchurl {
          url = "${release}/transcribe-native-${version}-linux-x86_64-cpu-vulkan.tar.gz";
          hash = "sha256-S1joDGPX8rUxtOhEuUwHA1k/SKW+av2Ck4nN2hShPt8=";
        };
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = [ pkgs.stdenv.cc.cc.lib ];
        installPhase = ''
          runHook preInstall
          rm libggml-vulkan.so
          install -Dm755 -t $out/lib *.so
          cp -r licenses $out/
          runHook postInstall
        '';
      };

      transcribe-cpp = py.buildPythonPackage {
        pname = "transcribe_cpp";
        inherit version;
        format = "wheel";
        src = pkgs.fetchurl {
          url = "${release}/transcribe_cpp-${version}-py3-none-any.whl";
          hash = "sha256-83O24gpNiX7Q3Kov5/rhgzeXd0MrOnZkHEJQbHe0Qq0=";
        };
        # The pip "native provider" package is replaced by TRANSCRIBE_LIBRARY.
        pythonRemoveDeps = [ "transcribe-cpp-native" ];
      };

      vad-model = pkgs.fetchurl {
        url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
        hash = "sha256-KZQNmNQrkfvQXOSJ8+z3xy8KQvAn5IdZGaKPtMBOos8=";
      };

      tapefront = py.buildPythonApplication {
        pname = "tapefront";
        version = "0.1.0";
        format = "other";
        src = self;
        dependencies = [ transcribe-cpp ];
        installPhase = ''
          runHook preInstall
          install -Dm755 tapefront.py $out/bin/tapefront
          runHook postInstall
        '';
        makeWrapperArgs = [
          "--prefix" "PATH" ":" (pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.whisper-cpp-vulkan ])
          "--set" "TRANSCRIBE_LIBRARY" "${transcribe-native}/lib/libtranscribe.so"
          "--set" "TAPEFRONT_VAD_MODEL" "${vad-model}"
        ];
        checkPhase = ''
          runHook preCheck
          python -m unittest -v test_tapefront
          runHook postCheck
        '';
        meta = {
          mainProgram = "tapefront";
          license = pkgs.lib.licenses.mit;
        };
      };
    in
    {
      packages.${system} = {
        inherit tapefront transcribe-native transcribe-cpp;
        default = tapefront;
      };
    };
}
