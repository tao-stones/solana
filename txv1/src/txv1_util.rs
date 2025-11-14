// Converts a legacy/v0 Transaction (the solana_transaction::Transaction produced
// by existing helpers in this file) into a serialized TxV1 byte vector per
// SIMD-0385. The produced bytes follow the layout:
//
// VersionByte (u8 = 129)
// LegacyHeader (3 x u8)
// NumInstructions (u8)
// TransactionConfigMask (u32 LE)  -- will be 0 for legacy->v1 conversion
// LifetimeSpecifier [u8;32]       -- uses the message recent_blockhash
// NumAddresses (u8) + Addresses ([u8;32])...
// ConfigValueSlots (none for conversion since mask == 0)
// InstructionHeaders (program_index:u8, num_accounts:u8, data_len:u16 LE)...
// InstructionPayloads (concat accounts bytes then data bytes)
// Signatures (num_required_signatures * 64 bytes)
//

use crate::txv1::*;
use solana_hash::Hash;
use solana_transaction::versioned::VersionedTransaction;

/// Construct an owned TransactionV1 from a legacy/v0 solana Transaction.
/// - TransactionConfigMask is set to 0 (we do not translate ComputeBudget instructions into config slots).
/// - LifetimeSpecifier is copied from the message recent_blockhash.
/// - All other fields are translated from the message/instructions/signatures.
pub fn from_legacy_transaction(tx: &VersionedTransaction) -> ResultT<TransactionV1> {
    let message = &tx.message;

    // Header
    let num_required_signatures = message.header().num_required_signatures;
    let num_readonly_signed_accounts = message.header().num_readonly_signed_accounts;
    let num_readonly_unsigned_accounts = message.header().num_readonly_unsigned_accounts;

    // Instructions
    let instructions = message.instructions();
    let num_instructions = instructions.len();
    if num_instructions > TransactionV1::MAX_INSTRUCTIONS {
        return Err(TxV1Error::TooManyInstructions(num_instructions));
    }

    // Addresses
    let account_keys = message.static_account_keys();
    let num_addresses = account_keys.len();
    if num_addresses > TransactionV1::MAX_ADDRESSES {
        return Err(TxV1Error::TooManyAddresses(num_addresses));
    }

    // Address count sanitization
    if num_addresses < (num_required_signatures as usize + num_readonly_unsigned_accounts as usize)
    {
        return Err(TxV1Error::InvalidAddressCount);
    }

    // Duplicate addresses check
    {
        let mut seen = std::collections::HashSet::new();
        for (i, pk) in account_keys.iter().enumerate() {
            if !seen.insert(pk) {
                return Err(TxV1Error::DuplicateAddress(i));
            }
        }
    }

    // Build addresses as [u8;32]
    let addresses: Vec<[u8; 32]> = account_keys.iter().map(|k| k.to_bytes()).collect();

    // Build instruction headers and payloads
    let mut instruction_headers: Vec<InstructionHeader> = Vec::with_capacity(num_instructions);
    let mut instruction_payloads: Vec<Vec<u8>> = Vec::with_capacity(num_instructions);

    for (i, inst) in instructions.iter().enumerate() {
        // Validate account indices
        for &idx in &inst.accounts {
            if (idx as usize) >= addresses.len() {
                return Err(TxV1Error::InstructionAccountIndexOutOfBounds {
                    instr: i,
                    index: idx,
                    num_addresses: addresses.len(),
                });
            }
        }
        if inst.accounts.len() > 255 {
            return Err(TxV1Error::InvalidSanitization(
                "too many accounts in instruction",
            ));
        }
        if inst.data.len() > u16::MAX as usize {
            return Err(TxV1Error::InvalidSanitization("instruction data too large"));
        }

        instruction_headers.push(InstructionHeader {
            program_account_index: inst.program_id_index,
            num_instruction_accounts: inst.accounts.len() as u8,
            num_instruction_data_bytes: inst.data.len() as u16,
        });

        let mut payload = Vec::with_capacity(inst.accounts.len() + inst.data.len());
        payload.extend_from_slice(&inst.accounts);
        payload.extend_from_slice(&inst.data);
        instruction_payloads.push(payload);
    }

    // TAO NOTE - copying signatures over for now since bench-tps tracks legacy transaction
    //            signature; It should be re-signed later.
    // Signatures: include first `num_required_signatures` signatures
    let sigs_expected = num_required_signatures as usize;
    if sigs_expected > TransactionV1::MAX_SIGNATURES {
        return Err(TxV1Error::TooManySignatures(sigs_expected));
    }
    let mut signatures: Vec<[u8; 64]> = Vec::with_capacity(sigs_expected);
    for i in 0..sigs_expected {
        let s = tx
            .signatures
            .get(i)
            .ok_or(TxV1Error::InvalidSanitization("missing signature"))?;
        signatures.push((*s).into());
    }

    // LifetimeSpecifier: recent_blockhash from message
    let recent_blockhash: Hash = *message.recent_blockhash();
    let lifetime_specifier: [u8; 32] = recent_blockhash.to_bytes();

    Ok(TransactionV1 {
        version: TransactionV1::VERSION_BYTE,
        legacy_header: LegacyHeader {
            num_required_signatures,
            num_readonly_signed_accounts,
            num_readonly_unsigned_accounts,
        },
        transaction_config_mask: 0u32, // no config slots when converting legacy->v1
        lifetime_specifier,
        num_instructions: num_instructions as u8,
        num_addresses: num_addresses as u8,
        addresses,
        config_value_slots: vec![],
        instruction_headers,
        instruction_payloads,
        signatures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_keypair::{Keypair, Signer};
    use solana_system_interface::instruction as system_instruction;
    use solana_transaction::{Message, Transaction};

    #[test]
    fn legacy_to_txv1_conversion_simple_transfer() {
        let from = Keypair::new();
        let to = Keypair::new();
        let lamports = 1u64;
        let recent_blockhash = Hash::new_unique();

        let transfer_ix = system_instruction::transfer(&from.pubkey(), &to.pubkey(), lamports);

        let message = Message::new(&[transfer_ix], Some(&from.pubkey()));
        let tx = Transaction::new(&[&from], message, recent_blockhash);

        let txv1 = from_legacy_transaction(&tx.clone().into()).expect("conversion should succeed");
        // basic checks
        assert_eq!(txv1.version, TransactionV1::VERSION_BYTE);
        assert_eq!(txv1.legacy_header.num_required_signatures, 1);
        assert_eq!(txv1.num_instructions, 1);
        assert_eq!(txv1.addresses.len(), tx.message().account_keys.len());
    }

    /// Convert a legacy Transaction into txv1 serialized bytes.
    /// - Sets VersionByte = 129
    /// - Leaves TransactionConfigMask == 0 (no config slots)
    /// - Copies recent_blockhash into LifetimeSpecifier
    /// - Copies account_keys, instructions, and signatures into txv1 layout
    pub fn legacy_transaction_to_txv1_bytes(tx: &Transaction) -> ResultT<Vec<u8>> {
        // safety limits taken from SIMD-0385
        const MAX_TRANSACTION_SIZE: usize = 4096;
        const MAX_INSTRUCTIONS: usize = 64;
        const MAX_ADDRESSES: usize = 64;
        const MAX_SIGNATURES: usize = 12;

        let message = tx.message();

        // Header values from the legacy message
        let num_required_signatures = message.header.num_required_signatures as usize;
        let _num_readonly_signed_accounts = message.header.num_readonly_signed_accounts as usize;
        let num_readonly_unsigned_accounts = message.header.num_readonly_unsigned_accounts as usize;

        // Instructions
        let instructions = &message.instructions;
        let num_instructions = instructions.len();
        if num_instructions > MAX_INSTRUCTIONS {
            return Err(TxV1Error::TooManyInstructions(num_instructions));
        }

        // Account keys (addresses)
        let account_keys = &message.account_keys;
        let num_addresses = account_keys.len();
        if num_addresses > MAX_ADDRESSES {
            return Err(TxV1Error::TooManyAddresses(num_addresses));
        }

        // Basic address count sanitization (spec: num_addresses >= num_required_signatures + num_readonly_unsigned_accounts)
        if num_addresses < num_required_signatures + num_readonly_unsigned_accounts {
            return Err(TxV1Error::InvalidAddressCount);
        }

        // Duplicate addresses check
        {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (i, pk) in account_keys.iter().enumerate() {
                if !seen.insert(pk) {
                    return Err(TxV1Error::DuplicateAddress(i));
                }
            }
        }

        // Signatures
        let sigs = &tx.signatures;
        if sigs.len() != num_required_signatures {
            // Some legacy transactions might contain a different signature count; enforce the invariant expected by SIMD-0385
            if sigs.len() > MAX_SIGNATURES {
                return Err(TxV1Error::TooManySignatures(sigs.len()));
            }
        }

        // Start serializing
        let mut out: Vec<u8> = Vec::with_capacity(512);

        // VersionByte
        out.push(129u8);

        // LegacyHeader (three u8)
        out.push(message.header.num_required_signatures);
        out.push(message.header.num_readonly_signed_accounts);
        out.push(message.header.num_readonly_unsigned_accounts);

        // TransactionConfigMask (u32 little-endian) -> 0 for conversion
        out.extend_from_slice(&0u32.to_le_bytes());

        // LifetimeSpecifier: Use recent_blockhash from the message
        // Message exposes recent_blockhash via `recent_blockhash()`
        // The Hash type implements AsRef<[u8;32]> via `as_ref()`
        let recent_blockhash = message.recent_blockhash;
        out.extend_from_slice(recent_blockhash.as_ref());

        // NumInstructions (u8)
        out.push(num_instructions as u8);

        // NumAddresses (u8)
        out.push(num_addresses as u8);

        // Addresses ([u8;32] each)
        for pk in account_keys.iter() {
            out.extend_from_slice(pk.as_ref());
        }

        // No config slots (mask == 0) -> nothing to append

        // InstructionHeaders
        for inst in instructions.iter() {
            // CompiledInstruction fields:
            //  - program_id_index: u8
            //  - accounts: Vec<u8>
            //  - data: Vec<u8>
            let program_index = inst.program_id_index;
            let num_accounts = inst.accounts.len();
            if num_accounts > 255 {
                return Err(TxV1Error::TooManyAddresses(num_accounts));
            }
            let data_len = inst.data.len();
            if data_len > u16::MAX as usize {
                return Err(TxV1Error::InvalidSanitization("InstructionDataTooLarge"));
            }
            out.push(program_index);
            out.push(num_accounts as u8);
            out.extend_from_slice(&(data_len as u16).to_le_bytes());
        }

        // InstructionPayloads: concatenated [account indices bytes][data bytes] per instruction
        for inst in instructions.iter() {
            // account indices
            out.extend_from_slice(&inst.accounts);
            // instruction data
            out.extend_from_slice(&inst.data);
        }

        // Signatures: only include the first `num_required_signatures` signatures
        for i in 0..(message.header.num_required_signatures as usize) {
            // signature type implements AsRef<[u8;64]> (or to_bytes())
            let sig_bytes = sigs.get(i).map(|s| s.as_ref()).unwrap_or(&[0u8; 64]); // in case of missing signature, fill zeros (shouldn't happen)
            out.extend_from_slice(sig_bytes);
        }

        // Final checks: size limit
        if out.len() > MAX_TRANSACTION_SIZE {
            return Err(TxV1Error::Other(
                "resulting txv1 exceeds max transaction size",
            ));
        }

        Ok(out)
    }

    #[test]
    fn test_convert_sample_transfer_to_txv1_wire_layout() {
        let from = Keypair::new();
        let to = Keypair::new();
        let lamports = 1u64;
        let recent_blockhash = Hash::new_unique();

        let transfer_ix = system_instruction::transfer(&from.pubkey(), &to.pubkey(), lamports);

        let message = Message::new(&[transfer_ix], Some(&from.pubkey()));
        let tx = Transaction::new(&[&from], message, recent_blockhash);

        let txv1 = from_legacy_transaction(&tx.clone().into()).expect("conversion should succeed");
        let expected_txv1_wire_layout = legacy_transaction_to_txv1_bytes(&tx)
            .expect("conversion to txv1 should succeed for simple transfer");

        // Basic sanity checks on produced expected_txv1_wire_layout
        // Version byte
        assert_eq!(expected_txv1_wire_layout[0], 129u8);

        // NumAddresses should equal message.account_keys().len()
        let message = tx.message();
        let num_addresses = message.account_keys.len();
        let idx_num_addresses = 1 /*version*/ + 3 /*legacy header*/ + 1 /*num_instructions*/ + 4 /*mask*/
            + 32 /*lifetime*/;
        assert_eq!(
            expected_txv1_wire_layout[idx_num_addresses],
            num_addresses as u8
        );

        // Size should be under the txv1 max
        assert!(expected_txv1_wire_layout.len() <= 4096);

        // wire layout should be same
        assert_eq!(txv1.serialize(), expected_txv1_wire_layout);
    }
}
