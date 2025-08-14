use {
    solana_account::AccountSharedData,
};

/// A trait to support chili peppers counter, used to identify is an account is hot/cold
pub trait AccountChiliPeppers {
    /// bank updates loaded accounts with its chili peppers counter during freezing
    fn update_chili_peppers(&self,
        chili_peppers: u64,
        // NOTE: dumb mocking - ideally AccountSharedData contains chili_peppers,
        // for prototype purpose, avoiding to change AccountSharedData, but asking
        // callsite to provide account_key to lookup for cached chili-peppers
        #[cfg(feature = "dev-context-only-utils")] account_key_for_mock: &Pubkey
        );

    /// read account's chili pepper count - the last time it was accessed
    fn read_chili_peppers(&self,
        #[cfg(feature = "dev-context-only-utils")] account_key_for_mock: &Pubkey
        ) -> u64;
}

// It is being debated writing chili peppers to accessed accounts - means to write
// to read-only accounts too.
// Mean while, for prototyping purpose, mocking account supports chili peppers
// with a local cache
use std::collections::HashMap;
use std::sync::RwLock;
use solana_pubkey::Pubkey;

// Local cache to track accounts' chili peppers
type ChiliPeppers = u64;
static ACCOUNTS_CHILI_PEPPERS: std::sync::LazyLock<RwLock<HashMap<Pubkey, ChiliPeppers>>> =
std::sync::LazyLock::new(|| {
    RwLock::new(HashMap::new())
});

impl AccountChiliPeppers for AccountSharedData {
    /// bank updates loaded accounts with its chili peppers counter during freezing
    fn update_chili_peppers(&self, chili_peppers: u64,
        #[cfg(feature = "dev-context-only-utils")] account_key_for_mock: &Pubkey
        ) {
        ACCOUNTS_CHILI_PEPPERS.write().unwrap().entry(*account_key_for_mock).and_modify(|v| *v = chili_peppers).or_insert(chili_peppers);
    }

    fn read_chili_peppers(&self,
        #[cfg(feature = "dev-context-only-utils")] account_key_for_mock: &Pubkey
        ) -> u64 {
        // if it's not in our cache, then it is a cold account
        *ACCOUNTS_CHILI_PEPPERS.read().unwrap().get(account_key_for_mock).unwrap_or(&0)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_account_chili_peppers() {
        let account_key = Pubkey::new_unique();
        let account = AccountSharedData::new(1, 2, &Pubkey::new_unique());

        // initially it is cold
        assert_eq!(account.read_chili_peppers(&account_key), 0);

        account.update_chili_peppers(101, &account_key);

        // now it has chilis
        assert_eq!(account.read_chili_peppers(&account_key), 101);
    }
}
