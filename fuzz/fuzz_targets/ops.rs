#![no_main]

use deallocate_zeroed_fuzzing::Ops;
use libfuzzer_sys::{fuzz_mutator, fuzz_target, fuzzer_mutate};
use mutatis::Session;

const ALLOCATION_LIMIT: usize = 1 << 20; // 1MiB

const fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

fuzz_mutator!(|data: &mut [u8], size: usize, max_size: usize, seed: u32| {
    let _ = env_logger::try_init();

    // With probability of about 1/8, just use the default mutator.
    if seed.count_ones() % 8 == 0 {
        return fuzzer_mutate(data, size, max_size);
    }

    // Decode the ops from the data, or use the default ops if that fails.
    let mut ops = bincode::decode_from_slice::<Ops, _>(data, bincode_config())
        .map_or_else(|_decode_err| Ops::default(), |(ops, _)| ops);

    let mut session = Session::new().seed(seed.into()).shrink(max_size < size);

    match session.mutate(&mut ops) {
        Ok(()) => {
            // Re-encode the mutated ops back into `data`.
            loop {
                if let Ok(new_size) = bincode::encode_into_slice(&ops, data, bincode_config()) {
                    return new_size;
                }

                // When re-encoding fails (presumably because `data` is not
                // large enough) then pop an op off the end and try again.
                if ops.pop() {
                    continue;
                }

                // But if we are out of ops to pop, then fall back to the
                // default fuzzer mutation strategy.
                break;
            }
        }
        Err(_) => {}
    }

    // If we failed to mutate the ops for whatever reason, fall back to the
    // fuzzer's default mutation strategies.
    return fuzzer_mutate(data, size, max_size);
});

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::try_init();
    if let Ok((ops, _)) = bincode::decode_from_slice::<Ops, _>(data, bincode_config()) {
        if let Err(e) = ops.run(ALLOCATION_LIMIT) {
            panic!("error: {e}");
        }
    }
});
