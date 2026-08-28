use {
    solana_clock::Epoch, solana_epoch_schedule::EpochSchedule,
    solana_genesis_config::GenesisConfig, solana_rent::Rent,
};

#[cfg_attr(feature = "frozen-abi", derive(solana_frozen_abi_macro::AbiExample))]
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RentCollector {
    pub epoch: Epoch,
    pub epoch_schedule: EpochSchedule,
    pub slots_per_year: f64,
    pub rent: Rent,
}

impl Default for RentCollector {
    fn default() -> Self {
        Self {
            epoch: Epoch::default(),
            epoch_schedule: EpochSchedule::default(),
            // derive default value using GenesisConfig::default()
            slots_per_year: GenesisConfig::default().slots_per_year(),
            rent: Rent::default(),
        }
    }
}

impl RentCollector {
    pub(crate) fn new(
        epoch: Epoch,
        epoch_schedule: EpochSchedule,
        slots_per_year: f64,
        rent: Rent,
    ) -> Self {
        Self {
            epoch,
            epoch_schedule,
            slots_per_year,
            rent,
        }
    }

    pub(crate) fn clone_with_epoch(&self, epoch: Epoch) -> Self {
        Self {
            epoch,
            ..self.clone()
        }
    }

    #[allow(deprecated)]
    pub(crate) fn deprecate_rent_exemption_threshold(&mut self) {
        let lamports_per_byte = match self.rent.exemption_threshold {
            threshold if threshold == 1.0f64.to_le_bytes() => self.rent.lamports_per_byte,
            threshold if threshold == 2.0f64.to_le_bytes() => self
                .rent
                .lamports_per_byte
                .checked_mul(2)
                .expect("SIMD-0194 rent threshold migration must not overflow"),
            // SIMD-0607 specifies integer-equivalent replay for the historical
            // mainnet thresholds. Custom genesis thresholds are left unchanged
            // rather than reintroducing consensus-critical float arithmetic.
            _ => self.rent.lamports_per_byte,
        };
        self.rent = Rent {
            lamports_per_byte,
            exemption_threshold: 1.0f64.to_le_bytes(),
            burn_percent: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    #[test]
    fn test_deprecate_rent_exemption_threshold_integer_equivalent_cases() {
        let epoch_schedule = EpochSchedule::default();

        let mut rent_collector = RentCollector::new(
            0,
            epoch_schedule.clone(),
            GenesisConfig::default().slots_per_year(),
            Rent {
                lamports_per_byte: 10,
                exemption_threshold: 1.0f64.to_le_bytes(),
                burn_percent: 25,
            },
        );
        rent_collector.deprecate_rent_exemption_threshold();
        assert_eq!(rent_collector.rent.lamports_per_byte, 10);
        assert_eq!(
            rent_collector.rent.exemption_threshold,
            1.0f64.to_le_bytes()
        );
        assert_eq!(rent_collector.rent.burn_percent, 50);

        let mut rent_collector = RentCollector::new(
            0,
            epoch_schedule,
            GenesisConfig::default().slots_per_year(),
            Rent {
                lamports_per_byte: 10,
                exemption_threshold: 2.0f64.to_le_bytes(),
                burn_percent: 25,
            },
        );
        rent_collector.deprecate_rent_exemption_threshold();
        assert_eq!(rent_collector.rent.lamports_per_byte, 20);
        assert_eq!(
            rent_collector.rent.exemption_threshold,
            1.0f64.to_le_bytes()
        );
        assert_eq!(rent_collector.rent.burn_percent, 50);
    }

    #[allow(deprecated)]
    #[test]
    fn test_deprecate_rent_exemption_threshold_custom_threshold() {
        let mut rent_collector = RentCollector::new(
            0,
            EpochSchedule::default(),
            GenesisConfig::default().slots_per_year(),
            Rent {
                lamports_per_byte: 10,
                exemption_threshold: 1.2f64.to_le_bytes(),
                burn_percent: 25,
            },
        );
        rent_collector.deprecate_rent_exemption_threshold();
        assert_eq!(rent_collector.rent.lamports_per_byte, 10);
        assert_eq!(
            rent_collector.rent.exemption_threshold,
            1.0f64.to_le_bytes()
        );
        assert_eq!(rent_collector.rent.burn_percent, 50);
    }
}
