//! TxV1 implementation (SIMD-0385) -- serialization, deserialization, config decoding,
//! and signing helpers.
//!
//! This module implements a parser/serializer for the Transaction V1 format and
//! adds:
//! - typed decoding of TransactionConfigMask -> ConfigFields (priority fee, compute units, etc).
//! - signing helpers using ed25519-dalek to sign and verify the transaction signing bytes.
//!
//! Behavior follows SIMD-0385:
//! - Version byte must be 129.
//! - Config slots: each set bit in TransactionConfigMask consumes a 4-byte slot, in increasing bit order.
//!   Priority-fee uses bits [0,1] (two slots -> 8 bytes LE as u64) and both bits must be set if present.
//! - Default values are used when bits are not set (priority_fee = 0, compute_unit_limit = 0, ...).
//!

use {
    solana_keypair::Keypair,
    solana_signature::Signature,
    solana_signer::Signer,
    std::{
        collections::HashSet,
        convert::TryInto,
        io::{Cursor, Read},
    },
};

/// TODO - to merge with TransactionError
#[derive(Debug)]
pub enum TxV1Error {
    UnexpectedEof,
    InvalidVersion(u8),
    TooManyInstructions(usize),
    TooManyAddresses(usize),
    TooManySignatures(usize),
    DuplicateAddress(usize),
    InvalidAddressCount,
    InstructionAccountIndexOutOfBounds {
        instr: usize,
        index: u8,
        num_addresses: usize,
    },
    TrailingData,
    InvalidSanitization(&'static str),
    SignatureVerificationFailed {
        idx: usize,
    },
    Other(&'static str),
}

type ResultT<T> = Result<T, TxV1Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyHeader {
    pub num_required_signatures: u8,
    pub num_readonly_signed_accounts: u8,
    pub num_readonly_unsigned_accounts: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstructionHeader {
    pub program_account_index: u8,
    pub num_instruction_accounts: u8,
    pub num_instruction_data_bytes: u16,
}

/// Typed config fields decoded from TransactionConfigMask + config slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigFields {
    /// Total lamports for transaction priority-fee (u64 LE). Default 0.
    pub priority_fee: u64,
    /// compute-unit-limit (u32 LE). Default 0.
    pub compute_unit_limit: u32,
    /// requested loaded accounts data size limit (u32 LE). Default 0.
    pub requested_accounts_data_size_limit: u32,
    /// requested heap size (u32 LE). Default 32768.
    pub requested_heap_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionV1 {
    pub version: u8, // must be 129
    pub legacy_header: LegacyHeader,
    pub transaction_config_mask: u32,
    pub lifetime_specifier: [u8; 32],
    pub num_instructions: u8,
    pub num_addresses: u8,
    pub addresses: Vec<[u8; 32]>,
    pub config_value_slots: Vec<[u8; 4]>,
    pub instruction_headers: Vec<InstructionHeader>,
    /// For each instruction, a payload: first the account indices bytes, then the instruction data bytes.
    pub instruction_payloads: Vec<Vec<u8>>,
    /// signatures[i] is signature for Addresses[i] signer (for i in 0 .. num_required_signatures)
    pub signatures: Vec<[u8; 64]>,
}

impl TransactionV1 {
    pub const VERSION_BYTE: u8 = 129;
    pub const MAX_TRANSACTION_SIZE: usize = 4096;
    pub const MAX_INSTRUCTIONS: usize = 64;
    pub const MAX_ADDRESSES: usize = 64;
    pub const MAX_SIGNATURES: usize = 12;
    pub const DEFAULT_HEAP_SIZE: u32 = 32_768;

    /// Serialize into bytes according to the spec layout. Does not enforce all sanitization rules
    /// beyond basic sizes; those are enforced in `deserialize_and_sanitize`.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = self.signing_bytes();

        // signatures
        for sig in &self.signatures {
            out.extend_from_slice(sig);
        }
        out
    }

    /// Returns the bytes that are signed: the serialization of the transaction up to (but not including) signatures.
    pub fn signing_bytes(&self) -> Vec<u8> {
        // Everything up to signatures.
        let mut out = Vec::new();
        out.push(self.version);
        out.push(self.legacy_header.num_required_signatures);
        out.push(self.legacy_header.num_readonly_signed_accounts);
        out.push(self.legacy_header.num_readonly_unsigned_accounts);
        out.extend_from_slice(&self.transaction_config_mask.to_le_bytes());
        out.extend_from_slice(&self.lifetime_specifier);
        out.push(self.num_instructions);
        out.push(self.num_addresses);
        for a in &self.addresses {
            out.extend_from_slice(a);
        }
        // config_value_slots: each is 4-bytes
        for slot in &self.config_value_slots {
            out.extend_from_slice(slot);
        }
        // instruction headers
        for ih in &self.instruction_headers {
            out.push(ih.program_account_index);
            out.push(ih.num_instruction_accounts);
            out.extend_from_slice(&ih.num_instruction_data_bytes.to_le_bytes());
        }
        // instruction payloads concatenated
        for payload in &self.instruction_payloads {
            out.extend_from_slice(payload);
        }
        out
    }

    /// Parse a TxV1 binary and run sanitization rules from the spec.
    pub fn deserialize_and_sanitize(bytes: &[u8]) -> ResultT<Self> {
        if bytes.len() > Self::MAX_TRANSACTION_SIZE {
            return Err(TxV1Error::InvalidSanitization("transaction too large"));
        }

        let read_u8 = |cur: &mut Cursor<&[u8]>| -> ResultT<u8> {
            let mut b = [0u8; 1];
            cur.read_exact(&mut b)
                .map_err(|_| TxV1Error::UnexpectedEof)?;
            Ok(b[0])
        };
        let read_bytes = |cur: &mut Cursor<&[u8]>, n: usize| -> ResultT<Vec<u8>> {
            let mut v = vec![0u8; n];
            cur.read_exact(&mut v)
                .map_err(|_| TxV1Error::UnexpectedEof)?;
            Ok(v)
        };

        // start parsing
        let mut cur = Cursor::new(bytes);
        let version = read_u8(&mut cur)?;
        if version != Self::VERSION_BYTE {
            return Err(TxV1Error::InvalidVersion(version));
        }
        let num_required_signatures = read_u8(&mut cur)?;
        let num_readonly_signed_accounts = read_u8(&mut cur)?;
        let num_readonly_unsigned_accounts = read_u8(&mut cur)?;
        let legacy_header = LegacyHeader {
            num_required_signatures,
            num_readonly_signed_accounts,
            num_readonly_unsigned_accounts,
        };
        // sanitize: require num_required_signatures >= 1 as implied by spec
        if legacy_header.num_required_signatures == 0 {
            return Err(TxV1Error::InvalidSanitization(
                "num_required_signatures must be >= 1",
            ));
        }
        // sanitization: num_readonly_signed_accounts < num_required_signatures
        if legacy_header.num_readonly_signed_accounts >= legacy_header.num_required_signatures {
            return Err(TxV1Error::InvalidSanitization(
                "num_readonly_signed_accounts >= num_required_signatures",
            ));
        }
        // read config mask (4 bytes LE)
        let mut mask_bytes = [0u8; 4];
        cur.read_exact(&mut mask_bytes)
            .map_err(|_| TxV1Error::UnexpectedEof)?;
        let transaction_config_mask = u32::from_le_bytes(mask_bytes);
        // lifetime specifier 32 bytes
        let lifetime_vec = read_bytes(&mut cur, 32)?;
        let lifetime_specifier: [u8; 32] = lifetime_vec.try_into().unwrap();
        // read instructions and addresses counters
        let num_instructions = read_u8(&mut cur)?;
        if (num_instructions as usize) > Self::MAX_INSTRUCTIONS {
            return Err(TxV1Error::TooManyInstructions(num_instructions as usize));
        }
        let num_addresses = read_u8(&mut cur)?;
        if (num_addresses as usize) > Self::MAX_ADDRESSES {
            return Err(TxV1Error::TooManyAddresses(num_addresses as usize));
        }
        // addresses
        let mut addresses = Vec::with_capacity(num_addresses as usize);
        for _ in 0..(num_addresses as usize) {
            let a = read_bytes(&mut cur, 32)?;
            addresses.push(a.try_into().unwrap());
        }
        // config value slots: one 4-byte slot per set bit in mask
        let popcount = transaction_config_mask.count_ones() as usize;
        let mut config_value_slots = Vec::with_capacity(popcount);
        for _ in 0..popcount {
            let s = read_bytes(&mut cur, 4)?;
            config_value_slots.push(s.try_into().unwrap());
        }

        // validate multi-bit fields rules: priority-fee uses bits 0 and 1 -> must be both set or both unset
        let bit0 = (transaction_config_mask & (1 << 0)) != 0;
        let bit1 = (transaction_config_mask & (1 << 1)) != 0;
        if bit0 ^ bit1 {
            return Err(TxV1Error::InvalidSanitization(
                "priority-fee bits [0,1] must both be set or both unset",
            ));
        }

        // instruction headers
        let mut instruction_headers = Vec::with_capacity(num_instructions as usize);
        for _ in 0..(num_instructions as usize) {
            let program_account_index = read_u8(&mut cur)?;
            let num_instruction_accounts = read_u8(&mut cur)?;
            let mut u16_bytes = [0u8; 2];
            cur.read_exact(&mut u16_bytes)
                .map_err(|_| TxV1Error::UnexpectedEof)?;
            let num_instruction_data_bytes = u16::from_le_bytes(u16_bytes);
            instruction_headers.push(InstructionHeader {
                program_account_index,
                num_instruction_accounts,
                num_instruction_data_bytes,
            });
        }
        // instruction payloads: iterate over headers and read payloads in order
        let mut instruction_payloads = Vec::with_capacity(num_instructions as usize);
        for (i, ih) in instruction_headers.iter().enumerate() {
            let accounts_len = ih.num_instruction_accounts as usize;
            let data_len = ih.num_instruction_data_bytes as usize;
            // read account indices
            let accounts_idx = read_bytes(&mut cur, accounts_len)?;
            // validate indices bounds
            for &idx in &accounts_idx {
                if idx >= num_addresses {
                    return Err(TxV1Error::InstructionAccountIndexOutOfBounds {
                        instr: i,
                        index: idx,
                        num_addresses: num_addresses as usize,
                    });
                }
            }
            let mut payload = Vec::with_capacity(accounts_len + data_len);
            payload.extend_from_slice(&accounts_idx);
            // read instruction data
            let data = read_bytes(&mut cur, data_len)?;
            payload.extend_from_slice(&data);
            instruction_payloads.push(payload);
        }
        // signatures: num_required_signatures of 64-bytes each
        let sigs_expected = legacy_header.num_required_signatures as usize;
        if sigs_expected > Self::MAX_SIGNATURES {
            return Err(TxV1Error::TooManySignatures(sigs_expected));
        }
        let mut signatures = Vec::with_capacity(sigs_expected);
        for _ in 0..sigs_expected {
            let s = read_bytes(&mut cur, 64)?;
            signatures.push(s.try_into().unwrap());
        }
        // no trailing data allowed
        let pos = cur.position() as usize;
        if pos != bytes.len() {
            return Err(TxV1Error::TrailingData);
        }
        // additional sanitization:
        // - num_addresses >= num_required_signatures + num_readonly_unsigned_accounts
        if (num_addresses as usize)
            < (legacy_header.num_required_signatures as usize
                + legacy_header.num_readonly_unsigned_accounts as usize)
        {
            return Err(TxV1Error::InvalidAddressCount);
        }
        // - duplicate addresses not allowed
        let mut seen = HashSet::new();
        for (i, a) in addresses.iter().enumerate() {
            if !seen.insert(a) {
                return Err(TxV1Error::DuplicateAddress(i));
            }
        }

        Ok(TransactionV1 {
            version,
            legacy_header,
            transaction_config_mask,
            lifetime_specifier,
            num_instructions,
            num_addresses,
            addresses,
            config_value_slots,
            instruction_headers,
            instruction_payloads,
            signatures,
        })
    }

    /// Decode the typed config fields from the mask + config slots, returning ConfigFields.
    ///
    /// This consumes the in-order config_value_slots mapping to the set bits of transaction_config_mask.
    /// If a requested field is not present (its bit(s) not set), default values are returned per spec.
    pub fn decode_config_fields(&self) -> ResultT<ConfigFields> {
        // popcount should equal slots length
        let popcount = self.transaction_config_mask.count_ones() as usize;
        if popcount != self.config_value_slots.len() {
            return Err(TxV1Error::InvalidSanitization(
                "config mask/popcount mismatch",
            ));
        }

        // default values
        let mut priority_fee: u64 = 0;
        let mut compute_unit_limit: u32 = 0;
        let mut requested_accounts_data_size_limit: u32 = 0;
        let mut requested_heap_size: u32 = Self::DEFAULT_HEAP_SIZE;

        // We'll iterate bits 0..31 and consume slots in order
        let mut slot_iter = self.config_value_slots.iter();

        // helper to read next slot as u32 LE
        let next_slot_u32 = |slot_iter: &mut std::slice::Iter<[u8; 4]>| -> ResultT<u32> {
            match slot_iter.next() {
                Some(slot) => Ok(u32::from_le_bytes(*slot)),
                None => Err(TxV1Error::InvalidSanitization("not enough config slots")),
            }
        };
        // read next slot 4 bytes as [u8;4]
        let next_slot_bytes = |slot_iter: &mut std::slice::Iter<[u8; 4]>| -> ResultT<[u8; 4]> {
            match slot_iter.next() {
                Some(slot) => Ok(*slot),
                None => Err(TxV1Error::InvalidSanitization("not enough config slots")),
            }
        };

        // iterate bits 0..=31 in increasing order
        for bit in 0usize..32 {
            if (self.transaction_config_mask & (1u32 << bit)) == 0 {
                continue; // bit unset, nothing consumed
            }
            match bit {
                0 => {
                    // bit 0 is part of priority-fee (needs bit1 too). Consume two slots
                    let low_bytes = next_slot_bytes(&mut slot_iter)?;
                    // consume slot for bit1 as well
                    let high_bytes = match slot_iter.next() {
                        Some(slot2) => *slot2,
                        None => {
                            return Err(TxV1Error::InvalidSanitization(
                                "not enough config slots for priority fee",
                            ))
                        }
                    };
                    let low_u64 = u64::from(u32::from_le_bytes(low_bytes));
                    let high_u64 = u64::from(u32::from_le_bytes(high_bytes));
                    priority_fee = (high_u64 << 32) | low_u64;
                }
                1 => {
                    // already consumed as part of bit 0 handling; skip
                    continue;
                }
                2 => {
                    compute_unit_limit = next_slot_u32(&mut slot_iter)?;
                }
                3 => {
                    requested_accounts_data_size_limit = next_slot_u32(&mut slot_iter)?;
                }
                4 => {
                    requested_heap_size = next_slot_u32(&mut slot_iter)?;
                }
                // future bits: consume slot but ignore (reserved)
                _ => {
                    let _ = next_slot_u32(&mut slot_iter)?;
                }
            }
        }

        Ok(ConfigFields {
            priority_fee,
            compute_unit_limit,
            requested_accounts_data_size_limit,
            requested_heap_size,
        })
    }

    /// Verify signatures for the transaction:
    /// - verifies signatures[0..num_required_signatures] using addresses[0..num_required_signatures] as public keys
    /// - returns Ok(()) if all verify, otherwise returns TxV1Error::SignatureVerificationFailed with failing index
    pub fn verify_signatures(&self) -> ResultT<()> {
        let n = self.legacy_header.num_required_signatures as usize;
        if self.signatures.len() != n {
            return Err(TxV1Error::InvalidSanitization("signature count mismatch"));
        }
        if self.addresses.len() < n {
            return Err(TxV1Error::InvalidSanitization(
                "addresses fewer than required signers",
            ));
        }

        let msg = self.signing_bytes();

        for i in 0..n {
            let pubkey_bytes = self.addresses[i];
            let sig = Signature::try_from(&self.signatures[i] as &[u8])
                .map_err(|_| TxV1Error::Other("invalid signature bytes"))?;
            if !sig.verify(&pubkey_bytes, &msg) {
                return Err(TxV1Error::SignatureVerificationFailed { idx: i });
            }
        }
        Ok(())
    }

    /// Sign the transaction using provided keypairs.
    /// The provided keypairs slice must correspond to the signers for addresses[0..num_required_signatures]
    /// (i.e., keypairs[i].public == addresses[i]).
    ///
    /// This replaces the signatures field with the generated signatures.
    pub fn sign_transaction(&mut self, signers: &[Keypair]) -> ResultT<()> {
        let n = self.legacy_header.num_required_signatures as usize;
        if signers.len() != n {
            return Err(TxV1Error::InvalidSanitization(
                "signers length must equal num_required_signatures",
            ));
        }
        if self.addresses.len() < n {
            return Err(TxV1Error::InvalidSanitization(
                "addresses fewer than required signers",
            ));
        }
        // ensure keypair public keys match addresses
        for i in 0..n {
            let pk_bytes = signers[i].pubkey().to_bytes();
            if pk_bytes != self.addresses[i] {
                return Err(TxV1Error::InvalidSanitization(
                    "signer public key does not match address",
                ));
            }
        }

        let msg = self.signing_bytes();
        let mut sigs: Vec<[u8; 64]> = Vec::with_capacity(n);
        for i in 0..n {
            let signature = signers[i].sign_message(&msg);
            sigs.push(signature.into());
        }
        self.signatures = sigs;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use {super::*, solana_pubkey::Pubkey};

    #[test]
    fn roundtrip_simple_transaction() {
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addresses = vec![
            Pubkey::new_unique().to_bytes(),
            Pubkey::new_unique().to_bytes(),
        ];
        let ih = InstructionHeader {
            program_account_index: 1,
            num_instruction_accounts: 1,
            num_instruction_data_bytes: 3,
        };
        let mut payload = Vec::new();
        payload.push(1u8);
        payload.extend_from_slice(&[0xAA, 0xBB, 0xCC]);

        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: 0,
            lifetime_specifier: [0u8; 32],
            num_instructions: 1,
            num_addresses: 2,
            addresses,
            config_value_slots: vec![],
            instruction_headers: vec![ih],
            instruction_payloads: vec![payload],
            signatures: vec![Signature::default().into()],
        };

        let ser = tx.serialize();
        assert!(ser.len() <= TransactionV1::MAX_TRANSACTION_SIZE);
        let parsed = TransactionV1::deserialize_and_sanitize(&ser).expect("parse ok");
        assert_eq!(parsed, tx);

        let signing_bytes = tx.signing_bytes();
        assert_eq!(&ser[0..signing_bytes.len()], signing_bytes.as_slice());
    }

    #[test]
    fn detect_duplicate_addresses() {
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addr = Pubkey::new_unique().to_bytes();
        let addresses = vec![addr, addr]; // duplicate
        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: 0,
            lifetime_specifier: [0u8; 32],
            num_instructions: 0,
            num_addresses: 2,
            addresses,
            config_value_slots: vec![],
            instruction_headers: vec![],
            instruction_payloads: vec![],
            signatures: vec![Signature::default().into()],
        };
        let ser = tx.serialize();
        let err = TransactionV1::deserialize_and_sanitize(&ser).unwrap_err();
        match err {
            TxV1Error::DuplicateAddress(_) => {}
            _ => panic!("expected DuplicateAddress, got {:?}", err),
        }
    }

    #[test]
    fn instruction_account_index_out_of_bounds() {
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addresses = vec![Pubkey::new_unique().to_bytes()]; // only 1 address => valid indices: 0
        let ih = InstructionHeader {
            program_account_index: 0,
            num_instruction_accounts: 1,
            num_instruction_data_bytes: 0,
        };
        let mut payload = Vec::new();
        payload.push(5u8); // invalid index (>= num_addresses)
        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: 0,
            lifetime_specifier: [0u8; 32],
            num_instructions: 1,
            num_addresses: 1,
            addresses,
            config_value_slots: vec![],
            instruction_headers: vec![ih],
            instruction_payloads: vec![payload],
            signatures: vec![Signature::default().into()],
        };
        let ser = tx.serialize();
        let err = TransactionV1::deserialize_and_sanitize(&ser).unwrap_err();
        match err {
            TxV1Error::InstructionAccountIndexOutOfBounds { .. } => {}
            _ => panic!("expected InstructionAccountIndexOutOfBounds, got {:?}", err),
        }
    }

    #[test]
    fn too_many_instructions_rejected() {
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addresses = vec![
            Pubkey::new_unique().to_bytes(),
            Pubkey::new_unique().to_bytes(),
        ];
        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: 0,
            lifetime_specifier: [0u8; 32],
            num_instructions: 65u8, // intentionally too large
            num_addresses: 2,
            addresses,
            config_value_slots: vec![],
            instruction_headers: vec![],
            instruction_payloads: vec![],
            signatures: vec![Signature::default().into()],
        };
        let ser = tx.serialize();
        let err = TransactionV1::deserialize_and_sanitize(&ser).unwrap_err();
        match err {
            TxV1Error::TooManyInstructions(65) => {}
            _ => panic!("expected TooManyInstructions, got {:?}", err),
        }
    }

    #[test]
    fn config_fields_decode_and_defaults() {
        let mask: u32 = 0b111;
        let low_slot = (0x5566_7788u32).to_le_bytes();
        let high_slot = (0x1122_3344u32).to_le_bytes();
        let compute_slot = (0xDEAD_BEEFu32).to_le_bytes();
        let config_slots = vec![low_slot, high_slot, compute_slot];
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addresses = vec![Pubkey::new_unique().to_bytes()];
        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: mask,
            lifetime_specifier: [0u8; 32],
            num_instructions: 0,
            num_addresses: 1,
            addresses,
            config_value_slots: config_slots.iter().map(|b| *b).collect(),
            instruction_headers: vec![],
            instruction_payloads: vec![],
            signatures: vec![Signature::default().into()],
        };
        let ser = tx.serialize();
        let parsed = TransactionV1::deserialize_and_sanitize(&ser).expect("parse ok");
        let cfg = parsed.decode_config_fields().expect("decode ok");
        let expected_priority = ((0x1122_3344u64) << 32) | (0x5566_7788u64);
        assert_eq!(cfg.priority_fee, expected_priority);
        assert_eq!(cfg.compute_unit_limit, 0xDEAD_BEEFu32);
        assert_eq!(cfg.requested_accounts_data_size_limit, 0);
        assert_eq!(cfg.requested_heap_size, TransactionV1::DEFAULT_HEAP_SIZE);
    }

    #[test]
    fn config_priority_bit_single_set_fails() {
        let mask: u32 = 1 << 0;
        let low_slot = (0x0102_0304u32).to_le_bytes();
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };
        let addresses = vec![Pubkey::new_unique().to_bytes()];
        let tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: mask,
            lifetime_specifier: [0u8; 32],
            num_instructions: 0,
            num_addresses: 1,
            addresses,
            config_value_slots: vec![low_slot],
            instruction_headers: vec![],
            instruction_payloads: vec![],
            signatures: vec![Signature::default().into()],
        };
        let ser = tx.serialize();
        let err = TransactionV1::deserialize_and_sanitize(&ser).unwrap_err();
        match err {
            TxV1Error::InvalidSanitization(_) => {}
            _ => panic!(
                "expected InvalidSanitization for priority bits, got {:?}",
                err
            ),
        }
    }

    #[test]
    fn sign_and_verify_transaction_with_keypairs() {
        let keypair = Keypair::new();

        let pk_bytes = keypair.pubkey().to_bytes();
        let legacy = LegacyHeader {
            num_required_signatures: 1,
            num_readonly_signed_accounts: 0,
            num_readonly_unsigned_accounts: 0,
        };

        let addresses = vec![pk_bytes, Pubkey::new_unique().to_bytes()];
        let mut tx = TransactionV1 {
            version: TransactionV1::VERSION_BYTE,
            legacy_header: legacy,
            transaction_config_mask: 0,
            lifetime_specifier: [0u8; 32],
            num_instructions: 0,
            num_addresses: 2,
            addresses,
            config_value_slots: vec![],
            instruction_headers: vec![],
            instruction_payloads: vec![],
            signatures: vec![[0u8; 64]], // placeholder
        };

        tx.sign_transaction(&[keypair]).expect("sign ok");
        tx.verify_signatures().expect("verify ok");

        // tamper signature -> verification should fail
        let mut bad_sig_tx = tx.clone();
        bad_sig_tx.signatures[0][0] ^= 0xFF;
        let err = bad_sig_tx.verify_signatures().unwrap_err();
        match err {
            TxV1Error::SignatureVerificationFailed { idx } => assert_eq!(idx, 0),
            _ => panic!("expected SignatureVerificationFailed, got {:?}", err),
        }
    }
}
