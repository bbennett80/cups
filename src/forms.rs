//! Guards and outcomes for the native POST → redirect/render form machines.
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FormState {
    Ready,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FormError {
    InvalidEmail,
    ShortPassword,
    PasswordMismatch,
    AccountExists,
    InvalidCredentials,
    InvalidClassName,
    ClassTaken,
}

impl FormError {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::InvalidEmail => "Enter a valid email.",
            Self::ShortPassword => "Password must be at least 8 characters.",
            Self::PasswordMismatch => "Passwords do not match.",
            Self::AccountExists => "A teacher account with this email already exists.",
            Self::InvalidCredentials => "Invalid email or password.",
            Self::InvalidClassName => {
                "Class URL must be 3-40 chars, lowercase letters/numbers/hyphens, and cannot start/end with a hyphen."
            }
            Self::ClassTaken => "That class URL is already taken. Choose another name.",
        }
    }
}

pub(crate) fn validate_signup(email: &str, password: &str, confirm: &str) -> Result<(), FormError> {
    if email.is_empty() || !email.contains('@') {
        Err(FormError::InvalidEmail)
    } else if password.chars().count() < 8 {
        Err(FormError::ShortPassword)
    } else if password != confirm {
        Err(FormError::PasswordMismatch)
    } else {
        Ok(())
    }
}

/// Only a validated name may reach the INSERT transition.
pub(crate) struct ClassName(String);

impl ClassName {
    pub(crate) fn parse(input: &str) -> Result<Self, FormError> {
        let name = input.trim().to_lowercase();
        if (3..=40).contains(&name.len())
            && name
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            && !name.starts_with('-')
            && !name.ends_with('-')
        {
            Ok(Self(name))
        } else {
            Err(FormError::InvalidClassName)
        }
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_names_match_the_form_contract() {
        for name in ["biology-101", "abc", "a-1", &"a".repeat(40)] {
            assert!(ClassName::parse(name).is_ok(), "{name}");
        }
        for name in [
            "a",
            "ab",
            "-abc",
            "abc-",
            "a b",
            "a/b",
            "éab",
            &"a".repeat(41),
        ] {
            assert!(ClassName::parse(name).is_err(), "{name}");
        }
        assert_eq!(
            ClassName::parse(" Biology-101 ").unwrap().as_str(),
            "biology-101"
        );
    }
}
