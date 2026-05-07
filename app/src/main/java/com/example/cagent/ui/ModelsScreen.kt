package com.example.cagent.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ModelsScreen(
    onOpenChat: () -> Unit,
    vm: ModelsViewModel = viewModel(),
) {
    val state by vm.state.collectAsState()
    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument(),
        onResult = { uri ->
            if (uri != null) vm.importModelFromUri(uri)
        }
    )

    LaunchedEffect(Unit) {
        vm.refresh()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(PaddingValues(20.dp)),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        Text("Models", style = MaterialTheme.typography.headlineMedium)
        Text(state.status, style = MaterialTheme.typography.bodyMedium)
        state.progressFraction?.let { LinearProgressIndicator(progress = { it }) }
        state.progressText?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

        Button(
            onClick = { picker.launch(arrayOf("*/*")) },
            enabled = !state.isBusy,
        ) {
            Text(if (state.isBusy) "Importing…" else "Import GGUF from device")
        }

        OutlinedButton(
            onClick = { vm.deleteImportedModel() },
            enabled = !state.isBusy && state.modelPath != null,
        ) {
            Text("Delete imported model")
        }

        OutlinedButton(
            onClick = onOpenChat,
            enabled = state.canOpenChat,
        ) {
            Text("Open chat")
        }

        Text(
            "Place a .gguf on the device (e.g. drag it into the emulator's Downloads), then tap “Import GGUF from device”.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

