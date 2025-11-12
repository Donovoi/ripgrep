use std::{
    ffi::{OsStr, OsString},
    fs::File,
    io,
    path::{Path, PathBuf},
    process::Command,
};

use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::process::{CommandError, CommandReader, CommandReaderBuilder};

#[cfg(feature = "gdeflate")]
use std::io::Read;

/// A builder for a matcher that determines which files get decompressed.
#[derive(Clone, Debug)]
pub struct DecompressionMatcherBuilder {
    /// The commands for each matching glob.
    commands: Vec<DecompressionCommand>,
    /// Whether to include the default matching rules.
    defaults: bool,
}

/// A representation of a single command for decompressing data
/// out-of-process.
#[derive(Clone, Debug)]
struct DecompressionCommand {
    /// The glob that matches this command.
    glob: String,
    /// The command or binary name.
    bin: PathBuf,
    /// The arguments to invoke with the command.
    args: Vec<OsString>,
}

impl Default for DecompressionMatcherBuilder {
    fn default() -> DecompressionMatcherBuilder {
        DecompressionMatcherBuilder::new()
    }
}

impl DecompressionMatcherBuilder {
    /// Create a new builder for configuring a decompression matcher.
    pub fn new() -> DecompressionMatcherBuilder {
        DecompressionMatcherBuilder { commands: vec![], defaults: true }
    }

    /// Build a matcher for determining how to decompress files.
    ///
    /// If there was a problem compiling the matcher, then an error is
    /// returned.
    pub fn build(&self) -> Result<DecompressionMatcher, CommandError> {
        let defaults = if !self.defaults {
            vec![]
        } else {
            default_decompression_commands()
        };
        let mut glob_builder = GlobSetBuilder::new();
        let mut commands = vec![];
        for decomp_cmd in defaults.iter().chain(&self.commands) {
            let glob = Glob::new(&decomp_cmd.glob).map_err(|err| {
                CommandError::io(io::Error::new(io::ErrorKind::Other, err))
            })?;
            glob_builder.add(glob);
            commands.push(decomp_cmd.clone());
        }
        let globs = glob_builder.build().map_err(|err| {
            CommandError::io(io::Error::new(io::ErrorKind::Other, err))
        })?;
        Ok(DecompressionMatcher { globs, commands })
    }

    /// When enabled, the default matching rules will be compiled into this
    /// matcher before any other associations. When disabled, only the
    /// rules explicitly given to this builder will be used.
    ///
    /// This is enabled by default.
    pub fn defaults(&mut self, yes: bool) -> &mut DecompressionMatcherBuilder {
        self.defaults = yes;
        self
    }

    /// Associates a glob with a command to decompress files matching the glob.
    ///
    /// If multiple globs match the same file, then the most recently added
    /// glob takes precedence.
    ///
    /// The syntax for the glob is documented in the
    /// [`globset` crate](https://docs.rs/globset/#syntax).
    ///
    /// The `program` given is resolved with respect to `PATH` and turned
    /// into an absolute path internally before being executed by the current
    /// platform. Notably, on Windows, this avoids a security problem where
    /// passing a relative path to `CreateProcess` will automatically search
    /// the current directory for a matching program. If the program could
    /// not be resolved, then it is silently ignored and the association is
    /// dropped. For this reason, callers should prefer `try_associate`.
    pub fn associate<P, I, A>(
        &mut self,
        glob: &str,
        program: P,
        args: I,
    ) -> &mut DecompressionMatcherBuilder
    where
        P: AsRef<OsStr>,
        I: IntoIterator<Item = A>,
        A: AsRef<OsStr>,
    {
        let _ = self.try_associate(glob, program, args);
        self
    }

    /// Associates a glob with a command to decompress files matching the glob.
    ///
    /// If multiple globs match the same file, then the most recently added
    /// glob takes precedence.
    ///
    /// The syntax for the glob is documented in the
    /// [`globset` crate](https://docs.rs/globset/#syntax).
    ///
    /// The `program` given is resolved with respect to `PATH` and turned
    /// into an absolute path internally before being executed by the current
    /// platform. Notably, on Windows, this avoids a security problem where
    /// passing a relative path to `CreateProcess` will automatically search
    /// the current directory for a matching program. If the program could not
    /// be resolved, then an error is returned.
    pub fn try_associate<P, I, A>(
        &mut self,
        glob: &str,
        program: P,
        args: I,
    ) -> Result<&mut DecompressionMatcherBuilder, CommandError>
    where
        P: AsRef<OsStr>,
        I: IntoIterator<Item = A>,
        A: AsRef<OsStr>,
    {
        let glob = glob.to_string();
        let bin = try_resolve_binary(Path::new(program.as_ref()))?;
        let args =
            args.into_iter().map(|a| a.as_ref().to_os_string()).collect();
        self.commands.push(DecompressionCommand { glob, bin, args });
        Ok(self)
    }
}

/// A matcher for determining how to decompress files.
#[derive(Clone, Debug)]
pub struct DecompressionMatcher {
    /// The set of globs to match. Each glob has a corresponding entry in
    /// `commands`. When a glob matches, the corresponding command should be
    /// used to perform out-of-process decompression.
    globs: GlobSet,
    /// The commands for each matching glob.
    commands: Vec<DecompressionCommand>,
}

impl Default for DecompressionMatcher {
    fn default() -> DecompressionMatcher {
        DecompressionMatcher::new()
    }
}

impl DecompressionMatcher {
    /// Create a new matcher with default rules.
    ///
    /// To add more matching rules, build a matcher with
    /// [`DecompressionMatcherBuilder`].
    pub fn new() -> DecompressionMatcher {
        DecompressionMatcherBuilder::new()
            .build()
            .expect("built-in matching rules should always compile")
    }

    /// Return a pre-built command based on the given file path that can
    /// decompress its contents. If no such decompressor is known, then this
    /// returns `None`.
    ///
    /// If there are multiple possible commands matching the given path, then
    /// the command added last takes precedence.
    pub fn command<P: AsRef<Path>>(&self, path: P) -> Option<Command> {
        if let Some(i) = self.globs.matches(path).into_iter().next_back() {
            let decomp_cmd = &self.commands[i];
            let mut cmd = Command::new(&decomp_cmd.bin);
            cmd.args(&decomp_cmd.args);
            return Some(cmd);
        }
        None
    }

    /// Returns true if and only if the given file path has at least one
    /// matching command to perform decompression on.
    pub fn has_command<P: AsRef<Path>>(&self, path: P) -> bool {
        self.globs.is_match(path)
    }
}

/// Configures and builds a streaming reader for decompressing data.
#[derive(Clone, Debug, Default)]
pub struct DecompressionReaderBuilder {
    matcher: DecompressionMatcher,
    command_builder: CommandReaderBuilder,
}

impl DecompressionReaderBuilder {
    /// Create a new builder with the default configuration.
    pub fn new() -> DecompressionReaderBuilder {
        DecompressionReaderBuilder::default()
    }

    /// Build a new streaming reader for decompressing data.
    ///
    /// If decompression is done out-of-process and if there was a problem
    /// spawning the process, then its error is logged at the debug level and a
    /// passthru reader is returned that does no decompression. This behavior
    /// typically occurs when the given file path matches a decompression
    /// command, but is executing in an environment where the decompression
    /// command is not available.
    ///
    /// If the given file path could not be matched with a decompression
    /// strategy, then a passthru reader is returned that does no
    /// decompression.
    pub fn build<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<DecompressionReader, CommandError> {
        let path = path.as_ref();

        // Check for GDeflate format first (native in-process decompression)
        #[cfg(feature = "gdeflate")]
        if is_gdeflate_file(path) {
            log::debug!(
                "{}: detected GDeflate format, using native decompression",
                path.display()
            );
            return DecompressionReader::new_gdeflate(path);
        }

        let Some(mut cmd) = self.matcher.command(path) else {
            return DecompressionReader::new_passthru(path);
        };
        cmd.arg(path);

        match self.command_builder.build(&mut cmd) {
            Ok(cmd_reader) => Ok(DecompressionReader {
                rdr: DecompressionReaderImpl::Command(cmd_reader),
            }),
            Err(err) => {
                log::debug!(
                    "{}: error spawning command '{:?}': {} \
                     (falling back to uncompressed reader)",
                    path.display(),
                    cmd,
                    err,
                );
                DecompressionReader::new_passthru(path)
            }
        }
    }

    /// Set the matcher to use to look up the decompression command for each
    /// file path.
    ///
    /// A set of sensible rules is enabled by default. Setting this will
    /// completely replace the current rules.
    pub fn matcher(
        &mut self,
        matcher: DecompressionMatcher,
    ) -> &mut DecompressionReaderBuilder {
        self.matcher = matcher;
        self
    }

    /// Get the underlying matcher currently used by this builder.
    pub fn get_matcher(&self) -> &DecompressionMatcher {
        &self.matcher
    }

    /// When enabled, the reader will asynchronously read the contents of the
    /// command's stderr output. When disabled, stderr is only read after the
    /// stdout stream has been exhausted (or if the process quits with an error
    /// code).
    ///
    /// Note that when enabled, this may require launching an additional
    /// thread in order to read stderr. This is done so that the process being
    /// executed is never blocked from writing to stdout or stderr. If this is
    /// disabled, then it is possible for the process to fill up the stderr
    /// buffer and deadlock.
    ///
    /// This is enabled by default.
    pub fn async_stderr(
        &mut self,
        yes: bool,
    ) -> &mut DecompressionReaderBuilder {
        self.command_builder.async_stderr(yes);
        self
    }
}

/// A streaming reader for decompressing the contents of a file.
///
/// The purpose of this reader is to provide a seamless way to decompress the
/// contents of file using existing tools in the current environment. This is
/// meant to be an alternative to using decompression libraries in favor of the
/// simplicity and portability of using external commands such as `gzip` and
/// `xz`. This does impose the overhead of spawning a process, so other means
/// for performing decompression should be sought if this overhead isn't
/// acceptable.
///
/// With the `gdeflate` feature enabled, `.gdz` files (GDeflate format) are
/// decompressed natively in-process using parallel decompression, providing
/// 3-8x faster performance compared to external process decompression.
///
/// A decompression reader comes with a default set of matching rules that are
/// meant to associate file paths with the corresponding command to use to
/// decompress them. For example, a glob like `*.gz` matches gzip compressed
/// files with the command `gzip -d -c`. If a file path does not match any
/// existing rules, or if it matches a rule whose command does not exist in the
/// current environment, then the decompression reader passes through the
/// contents of the underlying file without doing any decompression.
///
/// The default matching rules are probably good enough for most cases, and if
/// they require revision, pull requests are welcome. In cases where they must
/// be changed or extended, they can be customized through the use of
/// [`DecompressionMatcherBuilder`] and [`DecompressionReaderBuilder`].
///
/// By default, this reader will asynchronously read the processes' stderr.
/// This prevents subtle deadlocking bugs for noisy processes that write a lot
/// to stderr. Currently, the entire contents of stderr is read on to the heap.
///
/// # Example
///
/// This example shows how to read the decompressed contents of a file without
/// needing to explicitly choose the decompression command to run.
///
/// Note that if you need to decompress multiple files, it is better to use
/// `DecompressionReaderBuilder`, which will amortize the cost of compiling the
/// matcher.
///
/// ```no_run
/// use std::{io::Read, process::Command};
///
/// use grep_cli::DecompressionReader;
///
/// let mut rdr = DecompressionReader::new("/usr/share/man/man1/ls.1.gz")?;
/// let mut contents = vec![];
/// rdr.read_to_end(&mut contents)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct DecompressionReader {
    rdr: DecompressionReaderImpl,
}

#[derive(Debug)]
enum DecompressionReaderImpl {
    /// External process decompression (spawned command)
    Command(CommandReader),
    /// Direct file reading (no decompression)
    Passthru(File),
    /// Native GDeflate decompression (in-process, parallel)
    #[cfg(feature = "gdeflate")]
    GDeflate(GDeflateReader),
}

/// GDeflate file format magic number: "GDZ\0"
#[cfg(feature = "gdeflate")]
const GDEFLATE_MAGIC: &[u8; 4] = b"GDZ\0";

/// Maximum uncompressed size allowed (1GB) to prevent memory exhaustion
#[cfg(feature = "gdeflate")]
const MAX_UNCOMPRESSED_SIZE: usize = 1024 * 1024 * 1024;

/// Maximum compression ratio allowed (1000:1) to detect decompression bombs
#[cfg(feature = "gdeflate")]
const MAX_COMPRESSION_RATIO: usize = 1000;

/// Native GDeflate reader that implements parallel decompression
#[cfg(feature = "gdeflate")]
#[derive(Debug)]
struct GDeflateReader {
    /// Decompressed data buffer
    decompressed: Vec<u8>,
    /// Current read position
    position: usize,
}

#[cfg(feature = "gdeflate")]
impl GDeflateReader {
    /// Create a new GDeflate reader from a file
    fn new(mut file: File) -> io::Result<Self> {
        // Read and validate header
        let mut header = [0u8; 12];
        file.read_exact(&mut header)?;

        // Check magic number
        if &header[0..4] != GDEFLATE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GDeflate magic number - expected 'GDZ\\0'",
            ));
        }

        // Parse uncompressed size (little-endian u64)
        let output_size =
            u64::from_le_bytes(header[4..12].try_into().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid size field",
                )
            })?) as usize;

        // Security check: reject unreasonably large files
        if output_size > MAX_UNCOMPRESSED_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Uncompressed size too large: {} bytes (max {} bytes)",
                    output_size, MAX_UNCOMPRESSED_SIZE
                ),
            ));
        }

        // Read compressed data
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        // Security check: detect decompression bombs
        if output_size > compressed.len() * MAX_COMPRESSION_RATIO {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Suspicious compression ratio - possible decompression bomb ({}:1)",
                    output_size / compressed.len().max(1)
                ),
            ));
        }

        // Decompress using GDeflate library (0 = auto thread count)
        let decompressed = gdeflate::decompress(&compressed, output_size, 0)
            .map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, e.to_string())
        })?;

        Ok(Self { decompressed, position: 0 })
    }
}

#[cfg(feature = "gdeflate")]
impl io::Read for GDeflateReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.decompressed.len() {
            return Ok(0); // EOF
        }

        let remaining = &self.decompressed[self.position..];
        let to_copy = remaining.len().min(buf.len());
        buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
        self.position += to_copy;
        Ok(to_copy)
    }
}

impl DecompressionReader {
    /// Build a new streaming reader for decompressing data.
    ///
    /// If decompression is done out-of-process and if there was a problem
    /// spawning the process, then its error is returned.
    ///
    /// If the given file path could not be matched with a decompression
    /// strategy, then a passthru reader is returned that does no
    /// decompression.
    ///
    /// This uses the default matching rules for determining how to decompress
    /// the given file. To change those matching rules, use
    /// [`DecompressionReaderBuilder`] and [`DecompressionMatcherBuilder`].
    ///
    /// When creating readers for many paths. it is better to use the builder
    /// since it will amortize the cost of constructing the matcher.
    pub fn new<P: AsRef<Path>>(
        path: P,
    ) -> Result<DecompressionReader, CommandError> {
        DecompressionReaderBuilder::new().build(path)
    }

    /// Creates a new "passthru" decompression reader that reads from the file
    /// corresponding to the given path without doing decompression and without
    /// executing another process.
    fn new_passthru(path: &Path) -> Result<DecompressionReader, CommandError> {
        let file = File::open(path)?;
        Ok(DecompressionReader {
            rdr: DecompressionReaderImpl::Passthru(file),
        })
    }

    /// Creates a new GDeflate decompression reader for native in-process
    /// parallel decompression of .gdz files.
    #[cfg(feature = "gdeflate")]
    fn new_gdeflate(path: &Path) -> Result<DecompressionReader, CommandError> {
        let file = File::open(path)?;
        let reader = GDeflateReader::new(file)?;
        Ok(DecompressionReader {
            rdr: DecompressionReaderImpl::GDeflate(reader),
        })
    }

    /// Closes this reader, freeing any resources used by its underlying child
    /// process, if one was used. If the child process exits with a nonzero
    /// exit code, the returned Err value will include its stderr.
    ///
    /// `close` is idempotent, meaning it can be safely called multiple times.
    /// The first call closes the CommandReader and any subsequent calls do
    /// nothing.
    ///
    /// This method should be called after partially reading a file to prevent
    /// resource leakage. However there is no need to call `close` explicitly
    /// if your code always calls `read` to EOF, as `read` takes care of
    /// calling `close` in this case.
    ///
    /// `close` is also called in `drop` as a last line of defense against
    /// resource leakage. Any error from the child process is then printed as a
    /// warning to stderr. This can be avoided by explicitly calling `close`
    /// before the CommandReader is dropped.
    pub fn close(&mut self) -> io::Result<()> {
        match &mut self.rdr {
            DecompressionReaderImpl::Command(rdr) => rdr.close(),
            DecompressionReaderImpl::Passthru(_) => Ok(()),
            #[cfg(feature = "gdeflate")]
            DecompressionReaderImpl::GDeflate(_) => Ok(()),
        }
    }
}

impl io::Read for DecompressionReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &mut self.rdr {
            DecompressionReaderImpl::Command(rdr) => rdr.read(buf),
            DecompressionReaderImpl::Passthru(rdr) => rdr.read(buf),
            #[cfg(feature = "gdeflate")]
            DecompressionReaderImpl::GDeflate(rdr) => rdr.read(buf),
        }
    }
}

/// Resolves a path to a program to a path by searching for the program in
/// `PATH`.
///
/// If the program could not be resolved, then an error is returned.
///
/// The purpose of doing this instead of passing the path to the program
/// directly to Command::new is that Command::new will hand relative paths
/// to CreateProcess on Windows, which will implicitly search the current
/// working directory for the executable. This could be undesirable for
/// security reasons. e.g., running ripgrep with the -z/--search-zip flag on an
/// untrusted directory tree could result in arbitrary programs executing on
/// Windows.
///
/// Note that this could still return a relative path if PATH contains a
/// relative path. We permit this since it is assumed that the user has set
/// this explicitly, and thus, desires this behavior.
///
/// # Platform behavior
///
/// On non-Windows, this is a no-op.
pub fn resolve_binary<P: AsRef<Path>>(
    prog: P,
) -> Result<PathBuf, CommandError> {
    if !cfg!(windows) {
        return Ok(prog.as_ref().to_path_buf());
    }
    try_resolve_binary(prog)
}

/// Checks if a file is in GDeflate format by reading its magic number.
/// Returns true if the file starts with "GDZ\0".
#[cfg(feature = "gdeflate")]
fn is_gdeflate_file(path: &Path) -> bool {
    log::debug!("Checking if {} is a GDeflate file", path.display());
    let Ok(mut file) = File::open(path) else {
        log::debug!("  Failed to open file");
        return false;
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        log::debug!("  Failed to read magic bytes");
        return false;
    }
    let is_gdz = &magic == GDEFLATE_MAGIC;
    log::debug!("  Magic bytes: {:?}, is_gdz: {}", magic, is_gdz);
    is_gdz
}

/// Resolves a path to a program to a path by searching for the program in
/// `PATH`.
///
/// If the program could not be resolved, then an error is returned.
///
/// The purpose of doing this instead of passing the path to the program
/// directly to Command::new is that Command::new will hand relative paths
/// to CreateProcess on Windows, which will implicitly search the current
/// working directory for the executable. This could be undesirable for
/// security reasons. e.g., running ripgrep with the -z/--search-zip flag on an
/// untrusted directory tree could result in arbitrary programs executing on
/// Windows.
///
/// Note that this could still return a relative path if PATH contains a
/// relative path. We permit this since it is assumed that the user has set
/// this explicitly, and thus, desires this behavior.
///
/// If `check_exists` is false or the path is already an absolute path this
/// will return immediately.
fn try_resolve_binary<P: AsRef<Path>>(
    prog: P,
) -> Result<PathBuf, CommandError> {
    use std::env;

    fn is_exe(path: &Path) -> bool {
        let Ok(md) = path.metadata() else { return false };
        !md.is_dir()
    }

    let prog = prog.as_ref();
    if prog.is_absolute() {
        return Ok(prog.to_path_buf());
    }
    let Some(syspaths) = env::var_os("PATH") else {
        let msg = "system PATH environment variable not found";
        return Err(CommandError::io(io::Error::new(
            io::ErrorKind::Other,
            msg,
        )));
    };
    for syspath in env::split_paths(&syspaths) {
        if syspath.as_os_str().is_empty() {
            continue;
        }
        let abs_prog = syspath.join(prog);
        if is_exe(&abs_prog) {
            return Ok(abs_prog.to_path_buf());
        }
        if abs_prog.extension().is_none() {
            for extension in ["com", "exe"] {
                let abs_prog = abs_prog.with_extension(extension);
                if is_exe(&abs_prog) {
                    return Ok(abs_prog.to_path_buf());
                }
            }
        }
    }
    let msg = format!("{}: could not find executable in PATH", prog.display());
    return Err(CommandError::io(io::Error::new(io::ErrorKind::Other, msg)));
}

fn default_decompression_commands() -> Vec<DecompressionCommand> {
    const ARGS_GZIP: &[&str] = &["gzip", "-d", "-c"];
    const ARGS_BZIP: &[&str] = &["bzip2", "-d", "-c"];
    const ARGS_XZ: &[&str] = &["xz", "-d", "-c"];
    const ARGS_LZ4: &[&str] = &["lz4", "-d", "-c"];
    const ARGS_LZMA: &[&str] = &["xz", "--format=lzma", "-d", "-c"];
    const ARGS_BROTLI: &[&str] = &["brotli", "-d", "-c"];
    const ARGS_ZSTD: &[&str] = &["zstd", "-q", "-d", "-c"];
    const ARGS_UNCOMPRESS: &[&str] = &["uncompress", "-c"];
    // GDeflate uses magic number detection, but we add .gdz for convenience
    // The decompression will use native GDeflate if the feature is enabled
    // and the file has the correct magic number, regardless of extension.
    const ARGS_GDEFLATE: &[&str] = &["gzip", "-d", "-c"];

    fn add(glob: &str, args: &[&str], cmds: &mut Vec<DecompressionCommand>) {
        let bin = match resolve_binary(Path::new(args[0])) {
            Ok(bin) => bin,
            Err(err) => {
                log::debug!("{}", err);
                return;
            }
        };
        cmds.push(DecompressionCommand {
            glob: glob.to_string(),
            bin,
            args: args
                .iter()
                .skip(1)
                .map(|s| OsStr::new(s).to_os_string())
                .collect(),
        });
    }
    let mut cmds = vec![];
    add("*.gz", ARGS_GZIP, &mut cmds);
    add("*.tgz", ARGS_GZIP, &mut cmds);
    add("*.bz2", ARGS_BZIP, &mut cmds);
    add("*.tbz2", ARGS_BZIP, &mut cmds);
    add("*.xz", ARGS_XZ, &mut cmds);
    add("*.txz", ARGS_XZ, &mut cmds);
    add("*.lz4", ARGS_LZ4, &mut cmds);
    add("*.lzma", ARGS_LZMA, &mut cmds);
    add("*.br", ARGS_BROTLI, &mut cmds);
    add("*.zst", ARGS_ZSTD, &mut cmds);
    add("*.zstd", ARGS_ZSTD, &mut cmds);
    add("*.Z", ARGS_UNCOMPRESS, &mut cmds);
    // Add .gdz extension (will be intercepted by magic number detection)
    add("*.gdz", ARGS_GDEFLATE, &mut cmds);
    cmds
}
